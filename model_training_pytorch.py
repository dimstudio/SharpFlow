import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import preprocessing

import matplotlib.pyplot as plt
import numpy as np
import time
import data_helper


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, save_every: int = None, tensorboard: bool = False):
    if tensorboard:
        writer = SummaryWriter()
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_num = 0
        for xb, yb in train_dl:
            loss, num = loss_batch(model, loss_func, xb, yb, opt)
            train_loss += loss
            total_num += num
        train_loss /= total_num

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        acc, prec, recall = acc_prec_rec(model, valid_dl)
        print(f"Epoch: {epoch:5d}, Time: {(time.time()-start_time)/60:.3f} min, Train_loss: {train_loss:2.10f}, Val_loss: {val_loss:2.10f}, Accuracy: {acc:.5f}, Precision: {prec:.5f}, Recall: {recall:.5f}")
        # add to tensorboard
        if tensorboard:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/accuracy', acc, epoch)
            writer.add_scalar('Metrics/precision', prec, epoch)
            writer.add_scalar('Metrics/recall', recall, epoch)
        if save_every is not None:
            if epoch % save_every == 0:
                # save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, f"models/model_epoch_{epoch}.pt")


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


class MySmallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MySmallLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.lin = nn.Linear(in_features=self.hidden_size, out_features=output_size)

    def forward(self, x):
        out, state = self.lstm(x)
        out = self.lin(out[:, -1, :])
        out = torch.sigmoid(out)
        return out


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size//2, num_layers=1, batch_first=True)
        self.lin1 = nn.Linear(in_features=self.hidden_size//2, out_features=self.hidden_size//4)
        self.lin2 = nn.Linear(in_features=self.hidden_size//4, out_features=self.output_size)

    def forward(self, x):
        out, state = self.lstm1(x)
        out, state = self.lstm2(out)
        # Only take the last state of the second LSTM
        out = self.lin1(out[:, -1, :])
        out = torch.sigmoid(out)
        out = self.lin2(out)
        out = torch.sigmoid(out)
        return out


def get_data(folder, target_classes, to_exclude=None, ignore_files=None, dev="cpu", seed=1337):
    # np.random.seed(seed)
    sensor_data, annotations = data_helper.get_data_from_files(folder, ignore_files=ignore_files)
    ### Create tensor from files
    tensor = data_helper.tensor_transform(sensor_data, annotations, res_rate=25, to_exclude=to_exclude)
    # include only the relevant classes we are interested in
    targets = annotations[target_classes].values

    # Split into train, validation, test
    train_split = 0.7
    valid_split = 0.15
    test_split = 0.15
    perm_img_ind = np.random.permutation(range(len(tensor)))
    train_ind = perm_img_ind[:int(len(perm_img_ind) * train_split)]
    valid_ind = perm_img_ind[int(len(perm_img_ind) * train_split):int(len(perm_img_ind) * (train_split + valid_split))]
    test_ind = perm_img_ind[int(len(perm_img_ind) * (train_split + valid_split)):]

    x_train = np.array([tensor[i] for i in train_ind])
    y_train = np.array([targets[i] for i in train_ind])
    x_valid = np.array([tensor[i] for i in valid_ind])
    y_valid = np.array([targets[i] for i in valid_ind])
    x_test = np.array([tensor[i] for i in test_ind])
    y_test = np.array([targets[i] for i in test_ind])

    # Normalize/Scale only on train data. Use that scaler to later scale valid and test data
    # Pay attention to the range of your activation function! (Tanh --> [-1,1], Sigmoid --> [0,1])
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(x_train.reshape(x_train.shape[0] * x_train.shape[1], x_train.shape[2]))
    # Reshape sequences
    x_train = scaler.transform(x_train.reshape(x_train.shape[0] * x_train.shape[1], x_train.shape[2])).reshape(
        x_train.shape[0],
        x_train.shape[1],
        x_train.shape[2])
    x_valid = scaler.transform(x_valid.reshape(x_valid.shape[0] * x_valid.shape[1], x_valid.shape[2])).reshape(
        x_valid.shape[0],
        x_valid.shape[1],
        x_valid.shape[2])
    x_test = scaler.transform(x_test.reshape(x_test.shape[0] * x_test.shape[1], x_test.shape[2])).reshape(
        x_test.shape[0],
        x_test.shape[1],
        x_test.shape[2])

    x_train, y_train, x_valid, y_valid, x_test, y_test = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test)
    )

    batchsize = 64
    # Create as Dataset and use Dataloader
    train_ds = TensorDataset(x_train.float(), y_train.float())
    valid_ds = TensorDataset(x_valid.float(), y_valid.float())
    test_ds = TensorDataset(x_test.float(), y_test.float())
    # Create Dataloaders for the dataset
    train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batchsize * 2, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batchsize, shuffle=False)

    def putOnGPU(x, y):
        return x.to(dev), y.to(dev)

    train_dl = WrappedDataLoader(train_dl, putOnGPU)
    valid_dl = WrappedDataLoader(valid_dl, putOnGPU)
    test_dl = WrappedDataLoader(test_dl, putOnGPU)
    return train_dl, valid_dl, test_dl, x_train.shape[-1]


def train_model():
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # dev = "cpu"
    print(f"Device: {dev}")

    to_exclude = ['Ankle', 'Hip']  # variables to exclude
    ### Read Data
    # read data from session folder
    ignore_files = None  # ["kinect"]
    # folder = 'manual_sessions/tabletennis_strokes'
    # target_classes = ["correct_stroke"]
    folder = 'manual_sessions/cpr_experiment'
    target_classes = ['classRate', 'classDepth', 'classRelease']

    train_dl, valid_dl, test_dl, data_dim = get_data(folder=folder,
                                                     target_classes=target_classes,
                                                     to_exclude=to_exclude,
                                                     ignore_files=ignore_files,
                                                     dev=dev)
    # Input shape should be (batch_size, sequence_length, input_dimension)

    # Define model (done in function)
    lr = 0.01
    classes = len(target_classes)
    hidden_units = 128
    model = MyLSTM(data_dim, hidden_units, classes)
    # Put the model on GPU if available
    model.to(dev)
    # Define optimizer
    opt = optim.Adam(model.parameters(), lr=lr)
    # Loss function
    # loss_func = F.binary_cross_entropy  # use this loss only when class is binary
    loss_func = F.mse_loss
    # Training
    epochs = 100
    fit(epochs, model, loss_func, opt, train_dl, valid_dl, save_every=None, tensorboard=False)
    # Save model
    torch.save({'state_dict': model.state_dict()}, "models/lstm.pt")
    # Calculate accuracy
    acc, precision, recall = acc_prec_rec(model, test_dl)
    print(f"### Test set ### Accuracy: {acc:.5f} Precision: {precision:.5f} Recall: {recall:.5f}")


def acc_prec_rec(model, test_dl):
    # Accuracy for binary classification
    model.eval()
    with torch.no_grad():
        total_tp, total_tn, total_fp, total_fn = 0.0, 0.0, 0.0, 0.0
        for xb, yb in test_dl:
            ypred = model(xb)
            ypred_thresh = ypred > 0.5
            total_tp += torch.sum((ypred_thresh == 1) * (ypred_thresh == yb))
            total_tn += torch.sum((ypred_thresh == 0) * (ypred_thresh == yb))
            total_fp += torch.sum((ypred_thresh == 1) * (ypred_thresh != yb))
            total_fn += torch.sum((ypred_thresh == 0) * (ypred_thresh != yb))
        acc = (total_tp+total_tn)/(total_tp+total_tn+total_fn+total_fp)
        if total_tp+total_fp == 0:
            prec = 0
        else:
            prec = total_tp/(total_tp+total_fp)
        recall = total_tp/(total_tp+total_fn)
        return acc, prec, recall


def test_model(path_to_model):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # dev = "cpu"
    print(f"Device: {dev}")

    to_exclude = ['Ankle', 'Hip']  # variables to exclude
    ### Read Data
    # read data from session folder
    folder = 'manual_sessions/tabletennis_strokes'
    ignore_files = None
    # targetClasses = ['classRate', 'classDepth', 'classRelease', 'armsLocked', 'bodyWeight']
    target_classes = ["correct_stroke"]
    train_dl, valid_dl, test_dl, data_dim = get_data(folder=folder,
                                                     target_classes=target_classes,
                                                     to_exclude=to_exclude,
                                                     ignore_files=ignore_files,
                                                     dev=dev)
    # Input shape should be (batch_size, sequence_length, input_dimension)

    # Define model (done in function)
    classes = 1
    hidden_units = 128
    model = MySmallLSTM(data_dim, hidden_units, classes)
    model.load_state_dict(torch.load(path_to_model)["state_dict"])
    model.eval()
    # Test model with test data (fed in batches)
    acc, precision, recall = acc_prec_rec(model, test_dl)
    print(f"Accuracy: {acc:.5f} Precision: {precision:.5f} Recall: {recall:.5f}")


if __name__ == "__main__":
    train_model()
    # load_model()
    # test_model("models/lstm.pt")
