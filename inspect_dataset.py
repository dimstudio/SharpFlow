from utils.dataset_transportation import transportation_dataset
import numpy as np

# Shows number of samples in dataset and classes
data_folder = "manual_sessions/all_data/acc_magnitude"
dataset = transportation_dataset(data_path=data_folder, train=True, use_magnitude=True)
test_dataset = transportation_dataset(data_path=data_folder, train=False, use_magnitude=True)

print("Whole dataset size:", len(dataset))
existing_classes = np.unique(dataset.targets.numpy())
print(existing_classes)
for c in existing_classes:
    print("Class: ", c, " amount: ", np.sum(dataset.targets.numpy() == c))
print(dataset.targets.bincount())
