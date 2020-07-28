import os
import numpy as np
from utils.dataset_transportation import transportation_dataset
from train_transportationmode import need_train_test_folder, eval_model, classes
from utils import data_helper_transportation
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from metric.confusionmatrix import ConfusionMatrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report


def compare_classifiers(data_folder, model, batch_size, to_exclude=None, use_magnitude=False, valid_size=0.1, dev="cpu"):
    if to_exclude is None:
        sub_folder = "acc_magnitude" if use_magnitude else "all_sensors"
    else:
        sub_folder = "without_" + "_".join(to_exclude)
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
    # Dummy classifiers don't need train+val set
    data_for_dummy = (dataset.data-mean_train)/std_train
    test_dataset.data = (test_dataset.data - mean_train) / std_train
    test_dl = DataLoader(test_dataset, batch_size, shuffle=False,
                         num_workers=0, pin_memory=True)
    ###### Dummy Classifier 1 #######
    myStratifiedClassifier = DummyClassifier(strategy="stratified")
    myStratifiedClassifier.fit(data_for_dummy, dataset.targets)
    class_report = classification_report(myStratifiedClassifier.predict(test_dataset.data),
                                         test_dataset.targets,
                                         target_names=list(classes.keys()), output_dict=True, zero_division=0)
    metrics_str_stratClass = " ".join([
                                   f"{key} {class_report[key]['precision']:.2f}|{class_report[key]['recall']:.2f}|{class_report[key]['f1-score']:.2f}"
                                   for key in classes.keys()])
    print(f"StratDummy: Test_acc: {class_report['accuracy']:.2f}"
          f", Class-metrics (Precision|Recall|F1): {metrics_str_stratClass}")
    latex_str = " & ".join([
        f"{class_report[key]['precision']:.2f} & {class_report[key]['recall']:.2f} & {class_report[key]['f1-score']:.2f}"
        for key in classes.keys()])
    print(f"StratDummy latex: {latex_str}")
    ###### Dummy Classifier 2 ######
    myFrequentClassifier = DummyClassifier(strategy="most_frequent")
    myFrequentClassifier.fit(data_for_dummy, dataset.targets)
    class_report = classification_report(myFrequentClassifier.predict(test_dataset.data),
                                         test_dataset.targets,
                                         target_names=list(classes.keys()), output_dict=True, zero_division=0)
    metrics_str_freqClass = " ".join([
        f"{key} {class_report[key]['precision']:.2f}|{class_report[key]['recall']:.2f}|{class_report[key]['f1-score']:.2f}"
        for key in classes.keys()])
    print(f"FrequencyDummy: Test_acc: {class_report['accuracy']:.2f}"
          f", Class-metrics (Precision|Recall|F1): {metrics_str_freqClass}")
    latex_str = " & ".join([
                               f"{class_report[key]['precision']:.2f} & {class_report[key]['recall']:.2f} & {class_report[key]['f1-score']:.2f}"
                               for key in classes.keys()])
    print(f"FrequencyDummy latex: {latex_str}")

    ###### CNN ######
    loss_func = nn.CrossEntropyLoss().to(dev)
    # Use best NN-model for Test dataset:
    conf_mat = ConfusionMatrix(num_classes=6, normalized=False)
    test_loss, class_report = eval_model(model, loss_func, test_dl, conf_mat, dev=dev)
    # Use conf_mat to create metrics
    conf_mat = conf_mat.value()
    metrics_str_CNN = " ".join([f"{key} {class_report[key]['precision']:.2f}|{class_report[key]['recall']:.2f}|{class_report[key]['f1-score']:.2f}"
                               for key in classes.keys()])
    print(f"CNN: Test_acc: {class_report['accuracy']:.2f}"
          f", Class-metrics (Precision|Recall|F1): {metrics_str_CNN}")
    latex_str = " & ".join([f"{class_report[key]['precision']:.2f} & {class_report[key]['recall']:.2f} & {class_report[key]['f1-score']:.2f}"
                           for key in classes.keys()])
    print(f"CNN latex: {latex_str}")


if __name__ == "__main__":
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    best_model = "models/trained_models/TransportationCNN_without_Gyro_x_Gyro_y_Gyro_z.pt"
    loaded_model = torch.load(best_model)
    model = loaded_model["model"]
    model.load_state_dict(loaded_model["state_dict"])
    compare_classifiers(data_folder="manual_sessions/all_data",
                        model=model,
                        batch_size=1024,
                        use_magnitude=False,
                        to_exclude=["Gyro_x", "Gyro_y", "Gyro_z"],
                        dev=dev)