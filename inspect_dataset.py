from utils.dataset_transportation import transportation_dataset
import numpy as np

classes = {"Stationary": 0,
           "Walking": 1,
           "Running": 2,
           "Car": 3,
           "Train/Bus": 4,
           "Biking": 5
           }
# Shows number of samples in dataset and classes
data_folder = "manual_sessions/all_data/acc_magnitude"
dataset = transportation_dataset(data_path=data_folder, train=True, use_magnitude=True)
test_dataset = transportation_dataset(data_path=data_folder, train=False, use_magnitude=True)

print("Whole dataset size:", len(dataset))
existing_classes = np.unique(dataset.targets.numpy())
print(existing_classes)
for key, value in classes.items():
    print(f"Class:  {key:15} amount: {np.sum(dataset.targets.numpy() == value)}")
print(dataset.targets.bincount())
