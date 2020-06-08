import torch
from torch import nn
from utils import dataloader_transportation
from torchsummary import summary
from tqdm import tqdm
import time
import os
from models.Transportation_models import TransportationCNN


def train_model(data_folder, epochs, batch_size, learning_rate, earlystopping=None, save_every=None, dev="cpu"):
    # If needed create dataset from session files in data_folder

    # get the dataloaders (with the dataset)
    train_dl, valid_dl = dataloader_transportation.get_train_valid_loader(data_dir=data_folder,
                                                                          batch_size=batch_size,
                                                                          valid_size=0.1,
                                                                          shuffle=True,
                                                                          num_workers=0,
                                                                          pin_memory=True)
    # load the classification model
    model = TransportationCNN(n_classes=5)
    # Print the model and parameter count
    summary(model, (1, 13, 37), device="cpu")
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
        writer = SummaryWriter(comment=f"precip_{model.__class__.__name__}")
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
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in tqdm(valid_dl, desc="Validation", leave=False):
            # for xb, yb in valid_dl:
                loss = loss_func(model(xb.to(dev)), yb.to(dev))
                val_loss += loss.item()
            val_loss /= len(valid_dl)

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
              f"Train_loss: {train_loss:2.10f}, Val_loss: {val_loss:2.10f}"
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
    train_model(data_folder="data/some_cool_data",
                epochs=50,
                batch_size=64,
                learning_rate=0.01,
                earlystopping=30, dev=dev)
