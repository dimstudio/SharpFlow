import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle
import os
import time
import numpy as np

if __name__ == "__main__":
    data_folder = "manual_sessions/blackforest/acc_magnitude/"
    data_path = data_folder + "train"
    with open(os.path.join(data_path, "sensor_data.pkl"), "rb") as f:
        data_train = pickle.load(f)
    with open(os.path.join(data_path, "annotations.pkl"), "rb") as f:
        targets_train = pickle.load(f)

    # data_train = np.transpose(data_train, (0, 2, 1))
    start_time = time.time()
    # apply sklearn
    # Create the model with 100 trees
    model = RandomForestClassifier(n_estimators=100,
                                   bootstrap=True,
                                   max_features='sqrt')
    # Fit on training data
    model.fit(data_train, targets_train)
    print(f"Training took: {time.time() - start_time:.2f}s")
    # Test model on test set
    data_path = data_folder + "test"
    with open(os.path.join(data_path, "sensor_data.pkl"), "rb") as f:
        data_test = pickle.load(f)
    with open(os.path.join(data_path, "annotations.pkl"), "rb") as f:
        targets_test = pickle.load(f)
    rf_predictions = model.predict(data_test)
    rf_probs = model.predict_proba(data_test)
    roc_value = roc_auc_score(targets_test, rf_probs, multi_class="ovo")
    print("ROC_AUC: ", roc_value)
