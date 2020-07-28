from utils.dataset_transportation import transportation_dataset
import numpy as np

classes = {"Stationary": 0,
           "Walking": 1,
           "Running": 2,
           "Car": 3,
           "Train/Bus": 4,
           "Biking": 5
           }


def show_class_distribution(data_folder, use_magnitude):
    # Shows number of samples in dataset and classes
    # data_folder = "manual_sessions/all_data/acc_magnitude"
    dataset = transportation_dataset(data_path=data_folder, train=True, use_magnitude=use_magnitude)
    test_dataset = transportation_dataset(data_path=data_folder, train=False, use_magnitude=use_magnitude)

    print("Whole dataset size:", len(dataset)+len(test_dataset))
    existing_classes = np.unique(dataset.targets.numpy())
    # print(existing_classes)
    print("Class \t\t\t   Train     Test    Total   Proportion")
    print("-"*65)
    for key, value in classes.items():
        train_amount = np.sum(dataset.targets.numpy() == value)
        test_amount = np.sum(test_dataset.targets.numpy() == value)
        print(f"{key:15} amount: {train_amount:8} {test_amount:8} {train_amount+test_amount:8} {(train_amount+test_amount)/(len(dataset)+len(test_dataset))*100:10.2f}%")
    print("Train:", dataset.targets.bincount())
    print("Test:", test_dataset.targets.bincount())


if __name__ == "__main__":
    data_folder = "../manual_sessions/all_data/all_sensors"
    use_magnitude = False
    show_class_distribution(data_folder, use_magnitude)