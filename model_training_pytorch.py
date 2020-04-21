import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import preprocessing
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import time
import data_helper
import joblib


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
        f1 = 2 * prec * recall / (prec + recall)
        print(f"Epoch: {epoch:5d}, Time: {(time.time() - start_time) / 60:.3f} min, Train_loss: {train_loss:2.10f}, "
              f"Val_loss: {val_loss:2.10f}, Accuracy: {acc:.5f}, Precision: {prec:.5f}, Recall: {recall:.5f}, F1-Score: {f1}")
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
        self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size // 2, num_layers=1,
                             batch_first=True)
        self.lin1 = nn.Linear(in_features=self.hidden_size // 2, out_features=self.hidden_size // 4)
        self.lin2 = nn.Linear(in_features=self.hidden_size // 4, out_features=self.output_size)

    def forward(self, x):
        out, state = self.lstm1(x)
        out, state = self.lstm2(out)
        # Only take the last state of the second LSTM
        out = self.lin1(out[:, -1, :])
        out = torch.sigmoid(out)
        out = self.lin2(out)
        out = torch.sigmoid(out)
        return out


def load_train_data(train_folder, train_valid_split=0.7, to_exclude=None, ignore_files=None, target_classes=None,
                    batchsize=64, dev='cpu'):
    tensor_data, annotations = data_helper.get_data_from_files(train_folder, ignore_files=ignore_files, res_rate=25,
                                                               to_exclude=to_exclude)
    print("Shape tensor_data "+str(tensor_data.shape))
    print("Shape annotations " + str(annotations.shape))
    # Create tensor from files
    # tensor_data = data_helper.tensor_transform(sensor_data, annotations, res_rate=25, to_exclude=to_exclude)
    # include only the relevant classes we are interested in
    targets = annotations[target_classes].values

    # Split into train, validation
    perm_img_ind = np.random.permutation(range(len(tensor_data)))
    train_ind = perm_img_ind[:int(len(perm_img_ind) * train_valid_split)]
    valid_ind = perm_img_ind[int(len(perm_img_ind) * train_valid_split):]

    x_train = np.array([tensor_data[i] for i in train_ind])
    y_train = np.array([targets[i] for i in train_ind])
    x_valid = np.array([tensor_data[i] for i in valid_ind])
    y_valid = np.array([targets[i] for i in valid_ind])

    train_dl, valid_dl, data_dim, scaler = get_dataloader(x_train, y_train, x_valid, y_valid, batchsize, dev)
    return train_dl, valid_dl, data_dim, scaler


def get_dataloader(x_train, y_train, x_valid, y_valid, batchsize, dev):
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

    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )

    # Create as Dataset and use Dataloader
    train_ds = TensorDataset(x_train.float(), y_train.float())
    valid_ds = TensorDataset(x_valid.float(), y_valid.float())
    # Create Dataloaders for the dataset
    train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batchsize * 2, shuffle=False)

    def putOnGPU(x, y):
        return x.to(dev), y.to(dev)

    train_dl = WrappedDataLoader(train_dl, putOnGPU)
    valid_dl = WrappedDataLoader(valid_dl, putOnGPU)
    data_dim = x_train.shape[-1]
    return train_dl, valid_dl, data_dim, scaler


def load_test_data(test_folder, scaler=None, to_exclude=None, ignore_files=None, target_classes=None, batchsize=64,
                   dev='cpu'):
    tensor_data, annotations = data_helper.get_data_from_files(test_folder, ignore_files=ignore_files, res_rate=25,
                                                               to_exclude=to_exclude)
    ### Create tensor from files
    # tensor = data_helper.tensor_transform(sensor_data, annotations, res_rate=25, to_exclude=to_exclude)
    # include only the relevant classes we are interested in
    targets = annotations[target_classes].values

    x_test = tensor_data
    y_test = targets

    # Use the scaler from the training phase if specified
    # Pay attention to the range of your activation function! (Tanh --> [-1,1], Sigmoid --> [0,1])
    if scaler is not None:
        # Reshape sequences
        x_test = scaler.transform(x_test.reshape(x_test.shape[0] * x_test.shape[1], x_test.shape[2])).reshape(
            x_test.shape[0],
            x_test.shape[1],
            x_test.shape[2])

    x_test, y_test = map(
        torch.tensor, (x_test, y_test)
    )

    # Create as Dataset and use Dataloader
    test_ds = TensorDataset(x_test.float(), y_test.float())
    # Create Dataloaders for the dataset
    test_dl = DataLoader(test_ds, batch_size=batchsize, shuffle=False)

    def putOnGPU(x, y):
        return x.to(dev), y.to(dev)

    test_dl = WrappedDataLoader(test_dl, putOnGPU)
    data_dim = x_test.shape[-1]
    return test_dl, data_dim


def need_train_test_folder(dataset, ignore_files):
    # Data needs to be split into train and testing folder
    # use the data helper function to do this
    if ignore_files is None:
        ann_name = "annotations.pkl"
        sensor_name = "sensor_data.pkl"
    else:
        ann_name = f"annotations_ignorefiles{'_'.join(ignore_files)}.pkl"
        sensor_name = f"sensor_data_ignorefiles{'_'.join(ignore_files)}.pkl"

    return not os.path.isfile(f"{dataset}/train/{ann_name}") \
           or not os.path.isfile(f"{dataset}/train/{sensor_name}") \
           or not os.path.isfile(f"{dataset}/test/{ann_name}") \
           or not os.path.isfile(f"{dataset}/test/{sensor_name}")


def acc_prec_rec(model, test_dl):
    # Accuracy for BINARY classification
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
        acc = (total_tp + total_tn) / (total_tp + total_tn + total_fn + total_fp)
        if total_tp + total_fp == 0:
            prec = 0
        else:
            prec = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        return acc, prec, recall


def acc_prec_rec_modules(model, test_dl):
    # Accuracy for BINARY classification
    model.eval()
    with torch.no_grad():
        total_tp, total_tn, total_fp, total_fn = torch.zeros(1, model.output_size), torch.zeros(1,
                                                                                                model.output_size), torch.zeros(
            1, model.output_size), torch.zeros(1, model.output_size)
        for xb, yb in test_dl:
            ypred = model(xb)
            ypred_thresh = ypred > 0.5
            total_tp += torch.sum((ypred_thresh == 1) * (ypred_thresh == yb), dim=0)
            total_tn += torch.sum((ypred_thresh == 0) * (ypred_thresh == yb), dim=0)
            total_fp += torch.sum((ypred_thresh == 1) * (ypred_thresh != yb), dim=0)
            total_fn += torch.sum((ypred_thresh == 0) * (ypred_thresh != yb), dim=0)
        acc = (total_tp + total_tn) / (total_tp + total_tn + total_fn + total_fp)
        prec = torch.zeros(1, model.output_size)
        prec[total_tp + total_fp != 0] = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        return acc.squeeze(), prec.squeeze(), recall.squeeze()


def train_model(epochs, hidden_units, learning_rate, loss_function, batch_size=64, save_model_to='models/lstm',
                train_folder=None, to_exclude=None, ignore_files=None, target_classes=None):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # dev = "cpu"
    print(f"Device: {dev}")
    ### Load Data
    train_dl, valid_dl, data_dim, scaler = load_train_data(train_folder=train_folder,
                                                           train_valid_split=0.7,
                                                           to_exclude=to_exclude,
                                                           ignore_files=ignore_files,
                                                           target_classes=target_classes,
                                                           batchsize=batch_size,
                                                           dev=dev)
    # Save the scaler with the model name
    joblib.dump(scaler, f"{save_model_to}_scaler.pkl")
    # Input shape should be (batch_size, sequence_length, input_dimension)

    # Define model (done in function)
    classes = len(target_classes)
    model = MyLSTM(data_dim, hidden_units, classes)
    # Put the model on GPU if available
    model.to(dev)
    # Define optimizer
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    fit(epochs, model, loss_function, opt, train_dl, valid_dl, save_every=None, tensorboard=False)
    # Save model
    torch.save(dict(model=model, state_dict=model.state_dict()), f'{save_model_to}.pt')
    # torch.save({'state_dict': model.state_dict()}, f'{save_model_to}.pt')


def train_model_kfold(epochs, hidden_units, learning_rate, loss_function, batch_size=64, save_model_to='models/lstm',
                train_folder=None, to_exclude=None, ignore_files=None, target_classes=None):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # dev = "cpu"
    print(f"Device: {dev}")
    ### Load Data
    tensor_data, annotations = data_helper.get_data_from_files(train_folder, ignore_files=ignore_files, res_rate=25,
                                                               to_exclude=to_exclude)
    targets = annotations[target_classes].values
    kf = KFold(n_splits=10)
    kf.get_n_splits(tensor_data)
    KFold(n_splits=2, random_state=None, shuffle=False)
    for i, (train_index, test_index) in enumerate(kf.split(tensor_data)):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_valid = tensor_data[train_index], tensor_data[test_index]
        y_train, y_valid = targets[train_index], targets[test_index]

        train_dl, valid_dl, data_dim, scaler = get_dataloader(x_train, y_train,
                                                              x_valid, y_valid,
                                                              batch_size, dev)
        # Save the scaler with the model name
        joblib.dump(scaler, f"{save_model_to}_scaler.pkl")
        # Input shape should be (batch_size, sequence_length, input_dimension)

        # Define model (done in function)
        classes = len(target_classes)
        model = MyLSTM(data_dim, hidden_units, classes)
        # Put the model on GPU if available
        model.to(dev)
        # Define optimizer
        opt = optim.Adam(model.parameters(), lr=learning_rate)

        # Training
        fit(epochs, model, loss_function, opt, train_dl, valid_dl, save_every=None, tensorboard=False)
        # Save model
        torch.save(dict(model=model, state_dict=model.state_dict()), f'{save_model_to}_kfold_{i}.pt')
    # torch.save({'state_dict': model.state_dict()}, f'{save_model_to}.pt')


def test_model(path_to_model, test_folder=None, to_exclude=None, ignore_files=None, target_classes=None, batchsize=64):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # dev = "cpu"
    print(f"Device: {dev}")
    scaler = joblib.load(f"{path_to_model}_scaler.pkl")
    test_dl, data_dim = load_test_data(test_folder,
                                       scaler=scaler,
                                       to_exclude=to_exclude,
                                       ignore_files=ignore_files,
                                       target_classes=target_classes,
                                       batchsize=batchsize,
                                       dev=dev)
    # Input shape should be (batch_size, sequence_length, input_dimension)

    # Define model (done in function)
    loaded = torch.load(f'{path_to_model}.pt')
    model = loaded['model']
    model.load_state_dict(loaded['state_dict'])
    model.eval()
    model.to(dev)
    # Test model with test data (fed in batches)
    # Calculate accuracy, precision and recall
    # acc, precision, recall = acc_prec_rec(model, test_dl)
    acc, precision, recall = acc_prec_rec_modules(model, test_dl)
    f1 = 2 * precision * recall / (precision + recall)
    if len(target_classes) > 1:
        for i, tar_class in enumerate(target_classes):
            print(
                f"Target-class: {tar_class} Accuracy: {acc[i]:.5f} Precision: {precision[i]:.5f} Recall: {recall[i]:.5f} F1-Score: {f1[i]}")
    else:
        print(f"Accuracy: {acc:.5f} Precision: {precision:.5f} Recall: {recall:.5f} F1-Score: {f1}")


def train_test_model():
    dataset = "manual_sessions/CPR_feedback_binary"

    to_exclude = ['Ankle', 'Hip']  # variables to exclude
    ### Read Data
    # read data from session folder
    ignore_files = None  # ["kinect"]

    if need_train_test_folder(dataset, ignore_files):
        data_helper.create_train_test_folders(data=dataset,
                                              new_folder_location=None,
                                              train_test_ratio=0.85,
                                              ignore_files=ignore_files,
                                              to_exclude=to_exclude
                                              )

    # target_classes = ["correct_stroke"]
    target_classes = ['classRelease', 'classDepth', 'classRate', 'armsLocked', 'bodyWeight']

    save_model_to = "models/lstm"

    learning_rate = 0.01
    hidden_units = 128
    batch_size = 64
    epochs = 30
    # Loss function
    loss_func = F.binary_cross_entropy  # use this loss only when class is binary
    # loss_func = F.mse_loss
    train_model(epochs=epochs,
                learning_rate=learning_rate,
                hidden_units=hidden_units,
                batch_size=batch_size,
                loss_function=loss_func,
                save_model_to=save_model_to,
                train_folder=f"{dataset}/train",
                to_exclude=to_exclude,
                ignore_files=ignore_files,
                target_classes=target_classes)

    test_model(path_to_model=save_model_to,
               test_folder=f"{dataset}/test",
               to_exclude=to_exclude,
               ignore_files=ignore_files,
               target_classes=target_classes,
               batchsize=batch_size)


if __name__ == "__main__":
    train_test_model()
    # online_classification("models/lstm.pt")
