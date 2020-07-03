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
from metric.confusionmatrix import ConfusionMatrix
from sklearn.metrics import classification_report

classes = {"Stationary": 0,
           "Walking": 1,
           "Running": 2,
           "Car": 3,
           "Train/Bus": 4,
           "Biking": 5
           }


def need_train_test_folder(dataset):
    # Data needs to be split into train and testing folder
    # use the data helper function to do this
    ann_name = "annotations.pkl"
    sensor_name = "sensor_data.pkl"

    return not os.path.isfile(f"{dataset}/train/{ann_name}") \
           or not os.path.isfile(f"{dataset}/train/{sensor_name}") \
           or not os.path.isfile(f"{dataset}/test/{ann_name}") \
           or not os.path.isfile(f"{dataset}/test/{sensor_name}")


def eval_model(model, loss_func, valid_dl, conf_mat, dev="cpu"):
    val_loss = 0.0
    all_preds = []
    all_gts = []
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

            all_preds.append(pred_class.cpu().numpy())
            all_gts.append(yb.cpu().numpy())
            # Add prediction to confusion matrix
            conf_mat.add(pred_class, target=yb)
        val_loss /= len(valid_dl)
        class_report = classification_report(np.concatenate(all_gts), np.concatenate(all_preds),
                                             target_names=list(classes.keys()), output_dict=True, zero_division=0)

    return val_loss, class_report


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


def train_model(data_folder, epochs, n_classes, batch_size, learning_rate, valid_size=0.1, to_exclude=None, use_magnitude=True,
                earlystopping=None, lr_scheduler=True, save_every=None, dev="cpu"):
    sub_folder = "acc_magnitude" if use_magnitude else "all_sensors"
    # If needed create dataset from session files in data_folder
    if need_train_test_folder(os.path.join(data_folder, sub_folder)):
        data_helper_transportation.create_train_test_folders(data_folder, sub_folder=sub_folder,
                                                             to_exclude=to_exclude,
                                                             acc_magnitude=use_magnitude, smoothing=True)
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # load dataset
    dataset = transportation_dataset(data_path=os.path.join(data_folder, sub_folder), train=True, use_magnitude=use_magnitude)
    test_dataset = transportation_dataset(data_path=os.path.join(data_folder, sub_folder), train=False, use_magnitude=use_magnitude)
    # Split the data into training and validation set
    num_train = len(dataset)
    split_valid = int(np.floor(valid_size * num_train))
    split_train = num_train - split_valid
    train_dataset, valid_dataset = random_split(dataset, [split_train, split_valid])
    # Test dataset

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
        valid_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    test_dl = DataLoader(test_dataset, batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)



    # load the classification model
    input_channels = 1 if use_magnitude else dataset.data.shape[1]
    model = TransportationCNN(in_channels=input_channels, n_classes=n_classes, activation_function="elu", alpha=0.1)
    # Print the model and parameter count
    summary(model, (input_channels, 512), device="cpu")
    model.to(dev)
    # define optimizers and loss function
    # weight_decay is L2 weight normalization (used in paper), but I dont know how much
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.1, patience=5)
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
    for epoch in tqdm(range(epochs), desc="Epochs", leave=True, position=0):
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


        # Calc validation loss
        conf_mat = ConfusionMatrix(num_classes=n_classes, normalized=False)
        val_loss, class_report = eval_model(model, loss_func, valid_dl, conf_mat, dev=dev)
        # Use conf_mat to create metrics
        conf_mat = conf_mat.value()
        # per_class_acc = np.nan_to_num(conf_mat.diagonal()/conf_mat.sum(1))

        # Reduce learning rate after epoch
        if lr_scheduler:
            scheduler.step(val_loss)

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

        metrics_str = " ".join([f"{key} {class_report[key]['precision']:.2f}|{class_report[key]['recall']:.2f}|{class_report[key]['f1-score']:.2f}" for key in classes.keys()])
        print(f"Epoch: {epoch:5d}, Time: {(time.time() - start_time) / 60:.3f} min, "
              f"Train_loss: {train_loss:2.10f}, Val_loss: {val_loss:2.10f}, Val_acc: {class_report['accuracy']:.2f}"
              f", Class-metrics (Precision|Recall|F1): {metrics_str}"
              f", Early stopping counter: {earlystopping_counter}/{earlystopping}" if earlystopping is not None else "")

        if tensorboard:
            # add to tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            # TODO add confusion-matrix to tensorboard
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

    # Use best model for Test dataset:
    conf_mat = ConfusionMatrix(num_classes=n_classes, normalized=False)
    test_loss, class_report = eval_model(load_best_val_model, loss_func, test_dl, conf_mat, dev=dev)
    # Use conf_mat to create metrics
    conf_mat = conf_mat.value()
    metrics_str = " ".join([f"{key} {class_report[key]['precision']:.2f}|{class_report[key]['recall']:.2f}|{class_report[key]['f1-score']:.2f}"
                               for key in classes.keys()])
    print(f"Test_loss: {test_loss:2.10f}, Test_acc: {class_report['accuracy']:.2f}"
          f", Class-metrics (Precision|Recall|F1): {metrics_str}")


def test_model(data_folder, model, batch_size, to_exclude=None, use_magnitude=False, valid_size=0.1, dev="cpu"):
    sub_folder = "acc_magnitude" if use_magnitude else "all_sensors"
    # If needed create dataset from session files in data_folder
    if need_train_test_folder(os.path.join(data_folder, sub_folder)):
        data_helper_transportation.create_train_test_folders(data_folder, sub_folder=sub_folder,
                                                             to_exclude=to_exclude,
                                                             acc_magnitude=use_magnitude, smoothing=True)
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # load dataset
    dataset = transportation_dataset(data_path=os.path.join(data_folder, sub_folder), train=True,
                                     use_magnitude=use_magnitude)
    test_dataset = transportation_dataset(data_path=os.path.join(data_folder, sub_folder), train=False,
                                          use_magnitude=use_magnitude)
    # Split the data into training and validation set
    num_train = len(dataset)
    split_valid = int(np.floor(valid_size * num_train))
    split_train = num_train - split_valid
    train_dataset, valid_dataset = random_split(dataset, [split_train, split_valid])
    # Test dataset

    # normalize dataset (using scaler trained on training set)
    # get mean and std of trainset (for every feature)
    mean_train = torch.mean(train_dataset.dataset.data[train_dataset.indices], dim=0)
    std_train = torch.std(train_dataset.dataset.data[train_dataset.indices], dim=0)

    test_dataset.data = (test_dataset.data - mean_train) / std_train
    test_dl = DataLoader(test_dataset, batch_size, shuffle=False,
                         num_workers=0, pin_memory=True)

    loss_func = nn.CrossEntropyLoss().to(dev)
    # Use best model for Test dataset:
    conf_mat = ConfusionMatrix(num_classes=6, normalized=False)
    test_loss, class_report = eval_model(model, loss_func, test_dl, conf_mat, dev=dev)
    # Use conf_mat to create metrics
    conf_mat = conf_mat.value()
    metrics_str = " ".join([f"{key} {class_report[key]['precision']:.2f}|{class_report[key]['recall']:.2f}|{class_report[key]['f1-score']:.2f}"
                               for key in classes.keys()])
    print(f"Test_loss: {test_loss:2.10f}, Test_acc: {class_report['accuracy']:.2f}"
          f", Class-metrics (Precision|Recall|F1): {metrics_str}")


if __name__ == "__main__":
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # train_model(data_folder="manual_sessions/all_data",
    #             epochs=100,
    #             batch_size=1024,
    #             n_classes=6,
    #             learning_rate=0.0001,
    #             to_exclude=None,
    #             use_magnitude=False,
    #             earlystopping=30, dev=dev)

    best_model = "models/trained_models/TransportationCNN.pt"
    loaded_model = torch.load(best_model)
    model = loaded_model["model"]
    model.load_state_dict(loaded_model["state_dict"])
    test_model(data_folder="manual_sessions/all_data",
               model=model,
               batch_size=1024,
               use_magnitude=False,
               dev=dev)