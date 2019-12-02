import data_helper
# system imports
import os, datetime, time
import pickle
# ML imports
import numpy as np
# import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import logging
logging.getLogger('tensorflow').disabled = True


#####################################################

def model_training(input_tensor, targets, epochs=30, modelname=""):
    # Hyperparameters
    test_size = 0.33
    random_state = 88
    print("batch size: " + str(np.shape(input_tensor)) + " labels: " + str(np.shape(targets)))
    train_set, test_set, train_labels, test_labels = train_test_split(input_tensor, targets, test_size=test_size, random_state=random_state)
    input_tuple = (input_tensor.shape[1], input_tensor.shape[2])  # time-steps, data-dim
    hidden_dim = 32
    verbose = 2
    # make outputdim number of targets (and no softmax later on, because no classification)
    output_dim = targets.shape[1]
    # model definition
    print(('Keras model sequential target: ' + modelname))
    print(f"Output dimension: {output_dim}")
    # TODO Try stacking LSTMs
    # NOTE when stacking LSTMs you have to include return_sequences=True
    model = keras.Sequential([
        keras.layers.LSTM(units=hidden_dim, input_shape=input_tuple, return_sequences=True),
        keras.layers.LSTM(units=hidden_dim//2),
        keras.layers.Dense(16, activation='tanh'),
        keras.layers.Dense(output_dim, activation='tanh')
    ])

    # model compiling
    # TODO change loss to root_mean_squared_error, metrics to MSE
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='mse',
                  metrics=['mse'])
    # model fitting
    logdir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    model_history = model.fit(train_set, train_labels, validation_data=(test_set, test_labels), epochs=epochs,
                              verbose=verbose, callbacks=[tensorboard_callback])

    # Start Tensorboard in command line: tensorboard --logdir logs/
    # model storing
    store_model(model, 'models/model_' + modelname + '.h5')

    # y_prob = model.predict(test_set)
    test_loss, test_acc = model.evaluate(test_set, test_labels)
    print(('Test accuracy:', test_acc))
    print(('Test loss:', test_loss))
    # if target == 'classRelease':
    #     print(('Roc Auc', roc_auc_score(test_labels, y_prob.argmax(axis=1))))

    # Don't know what this is for...
    # y_true = pd.Series(test_labels)
    # y_pred = pd.Series(y_prob.argmax(axis=1))
    # pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# function
# IN: keras.model and path/to/model.h5
def store_model(the_model, path_model):
    print(('Saved model path: ' + path_model))
    the_model.save(path_model)


def load_model(target_name):
    filename = 'models/model_' + target_name + '.h5'
    if os.path.isfile(filename):
        new_model = keras.models.load_model(filename)
    else:
        print((filename + ' not found'))
    return new_model


def main():
    #####################################################
    ### VARIABLES
    # current_time_offset = 1
    to_exclude = ['Ankle', 'Hip', 'Hand']  # variables to exclude
    ### Read Data
    # read data from session folder
    folder = 'manual_sessions/tabletennis_strokes'
    ignoreKinect = True
    # targetClasses = ['classRate', 'classDepth', 'classRelease', 'armsLocked', 'bodyWeight']
    target_classes = ["correct_stroke"]
    sensor_data, annotations = data_helper.get_data_from_files(folder, ignoreKinect=ignoreKinect)
    ### Create tensor from files
    tensor = data_helper.tensor_transform(sensor_data, annotations, res_rate=25, to_exclude=to_exclude)
    # include only the relevant classes we are interested in
    targets = annotations[target_classes].values
    ### Create Training and validation set (and maybe even test?)
    test_size = 0.33
    random_state = 88
    print("batch size: " + str(np.shape(tensor)) + " labels: " + str(np.shape(targets)))
    train_set, test_set, train_labels, test_labels = train_test_split(tensor, targets, test_size=test_size,
                                                                      random_state=random_state)
    # Check/Remove outliers?
    # Data preprocessing - scaling the attributes
    # scaler.fit() should only be done on training set and then applied to the test set
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # Reshape sequences
    train_set = scaler.fit_transform(train_set.reshape(train_set.shape[0] * train_set.shape[1], train_set.shape[2])).reshape(train_set.shape[0], train_set.shape[1], train_set.shape[2])
    test_set = scaler.fit_transform(test_set.reshape(test_set.shape[0] * test_set.shape[1], test_set.shape[2])).reshape(test_set.shape[0], test_set.shape[1], test_set.shape[2])

    ### Train model
    modelname = "_".join(target_classes)
    epochs = 120

    input_tuple = (train_set.shape[1], train_set.shape[2])  # time-steps, data-dim
    hidden_dim = 32
    verbose = 2
    # make outputdim number of targets (and no softmax later on, because no classification)
    output_dim = targets.shape[1]
    # model definition
    print(('Keras model sequential target: ' + modelname))
    print(f"Output dimension: {output_dim}")
    # TODO Try stacking LSTMs
    # NOTE when stacking LSTMs you have to include return_sequences=True
    model = keras.Sequential([
        keras.layers.LSTM(units=hidden_dim, input_shape=input_tuple, return_sequences=True),
        keras.layers.LSTM(units=hidden_dim // 2),
        keras.layers.Dense(16, activation='tanh'),
        keras.layers.Dense(output_dim, activation='sigmoid')
    ])

    # model compiling
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='mse',
                  metrics=['mse'])
    # model fitting
    logdir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    model_history = model.fit(train_set, train_labels, validation_data=(test_set, test_labels), epochs=epochs,
                              verbose=verbose, callbacks=[tensorboard_callback])

    # Start Tensorboard in command line: tensorboard --logdir logs/
    # model storing
    store_model(model, 'models/model_' + modelname + '.h5')

    # y_prob = model.predict(test_set)
    test_loss, test_acc = model.evaluate(test_set, test_labels)
    print(('Test accuracy:', test_acc))
    print(('Test loss:', test_loss))


if __name__ == "__main__":
    main()
    # test_model("correct_stroke")
