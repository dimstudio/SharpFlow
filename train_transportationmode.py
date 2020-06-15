import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from utils import data_helper_transportation
from utils.dataset_transportation import transportation_dataset
from torchsummary import summary
from tqdm import tqdm
import time
import os
import numpy as np
from models.Transportation_models import TransportationCNN
from sklearn.preprocessing import StandardScaler


def need_train_test_folder(dataset):
    # Data needs to be split into train and testing folder
    # use the data helper function to do this
    ann_name = "annotations.pkl"
    sensor_name = "sensor_data.pkl"

    return not os.path.isfile(f"{dataset}/train/{ann_name}") \
           or not os.path.isfile(f"{dataset}/train/{sensor_name}") \
           or not os.path.isfile(f"{dataset}/test/{ann_name}") \
           or not os.path.isfile(f"{dataset}/test/{sensor_name}")


def eval_model(model, loss_func, valid_dl, dev="cpu"):
    val_loss = 0.0
    acc = 0.0
    model.eval()
    with torch.no_grad():
        # TODO calculate other metrics
        for xb, yb in tqdm(valid_dl, desc="Validation", leave=False):
            # for xb, yb in valid_dl:
            xb = xb.to(dev)
            yb = yb.to(dev)
            y_pred = model(xb)
            loss = loss_func(y_pred, yb)
            val_loss += loss.item()
            pred_class = torch.argmax(torch.log_softmax(y_pred, dim=1), dim=1)
            # Calculate Accuracy for all classes (including stationary)
            correct_pred = (pred_class == yb).float()
            acc += correct_pred.sum() / len(correct_pred)
            # Calculate acc without "Stationary"
            # acc += 1.0 * torch.sum((pred_class == yb) * (yb > 0)) / torch.sum(yb > 0)
        val_loss /= len(valid_dl)
        acc /= len(valid_dl)

    return val_loss, acc


def get_mean_std_online(loader):
    n_samples = 0
    mean = torch.empty(6)
    var = torch.empty(6)
    for i_batch, batch_target in enumerate(loader):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        n_samples += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        var += batch.var(2).sum(0)

    mean /= n_samples
    var /= n_samples
    std = torch.sqrt(var)
    return mean, std


def train_model(data_folder, epochs, batch_size, learning_rate, valid_size=0.1, earlystopping=None, save_every=None, dev="cpu"):
    # If needed create dataset from session files in data_folder
    if need_train_test_folder(data_folder):
        data_helper_transportation.create_train_test_folders(data_folder, to_exclude=None)
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # load dataset
    dataset = transportation_dataset(data_path=data_folder, train=True)
    # Split the data into training and validation set
    num_train = len(dataset)
    split_valid = int(np.floor(valid_size * num_train))
    split_train = num_train - split_valid
    train_dataset, valid_dataset = random_split(dataset, [split_train, split_valid])
    # Test dataset
    test_dataset = transportation_dataset(data_path=data_folder, train=False)

    # normalize dataset (using scaler trained on training set)
    # get mean and std of trainset (for every feature)
    mean_train = torch.mean(train_dataset.dataset.data[train_dataset.indices], dim=0)
    std_train = torch.std(train_dataset.dataset.data[train_dataset.indices], dim=0)
    train_dataset.dataset.data[train_dataset.indices] = (train_dataset.dataset.data[train_dataset.indices] - mean_train) / std_train
    valid_dataset.dataset.data[valid_dataset.indices] = (valid_dataset.dataset.data[valid_dataset.indices] - mean_train) / std_train
    test_dataset.data = (test_dataset.data - mean_train) / std_train
    # get the dataloaders (with the datasets)
    train_dl = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    valid_dl = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    test_dl = DataLoader(test_dataset, batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)



    # load the classification model
    model = TransportationCNN(in_channels=6, n_classes=6)
    # Print the model and parameter count
    summary(model, (6, 512), device="cpu")
    model.to(dev)
    # define optimizers and loss function
    # weight_decay is L2 weight normalization (used in paper), but I dont know how much
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.9)
    loss_func = nn.CrossEntropyLoss().to(dev)
    # fit the model
    tensorboard = False
    #### Training ####

    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(comment=f"transportation_{model.__class__.__name__}")
    start_time = time.time()
    best_val_loss = 1e300
    earlystopping_counter = 0
    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
        model.train()
        train_loss = 0.0
        for i, (xb, yb) in enumerate(tqdm(train_dl, desc="Batches", leave=False)):
        # for i, (xb, yb) in enumerate(train_dl):
            loss = loss_func(model(xb.to(dev)), yb.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            # if i > 100:
            #     break
        train_loss /= len(train_dl)

        # Reduce learning rate after epoch
        # scheduler.step()

        # Calc validation loss
        val_loss, acc = eval_model(model, loss_func, valid_dl, dev=dev)

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            os.makedirs("models/checkpoints", exist_ok=True)
            torch.save({
                'model': model,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, f"models/checkpoints/best_val_loss_model_{model.__class__.__name__}.pt")
            best_val_loss = val_loss
            earlystopping_counter = 0

        else:
            if earlystopping is not None:
                earlystopping_counter += 1
                if earlystopping_counter >= earlystopping:
                    print(f"Stopping early --> val_loss has not decreased over {earlystopping} epochs")
                    break

        print(f"Epoch: {epoch:5d}, Time: {(time.time() - start_time) / 60:.3f} min, "
              f"Train_loss: {train_loss:2.10f}, Val_loss: {val_loss:2.10f}, Acc: {acc:.2f}"
              f", Early stopping counter: {earlystopping_counter}/{earlystopping}" if earlystopping is not None else "")

        if tensorboard:
            # add to tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
        if save_every is not None:
            if epoch % save_every == 0:
                # save model
                torch.save({
                    'model': model,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }, f"models/checkpoints/model_{model.__class__.__name__}_epoch_{epoch}.pt")

    # Save best model
    load_best_val_model = torch.load(f"models/checkpoints/best_val_loss_model_{model.__class__.__name__}.pt")
    os.makedirs("models/trained_models", exist_ok=True)
    torch.save({'model': load_best_val_model['model'],
                'state_dict': load_best_val_model['state_dict']},
               f"models/trained_models/{model.__class__.__name__}.pt")


if __name__ == "__main__":
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_model(data_folder="manual_sessions/blackforest",
                epochs=50,
                batch_size=1024,
                learning_rate=0.01,
                earlystopping=30, dev=dev)
