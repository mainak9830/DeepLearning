import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"
LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32
classes = 10
def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return X, y
def get_data(data_path, test_size=0.1, test_validation=0.1):

    #load dataset
    X, y = load_dataset(data_path)

    #create train/validation/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_validation)

    #convert inputs from 2d to 3d
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):


    #build network
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu", input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(classes, activation="softmax"))
    #compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=error, metrics=["accuracy"])

    print(model.summary())
    return model

def main():

    # load train/validation/test data splits
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = get_data(DATA_PATH)

    #build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape, LEARNING_RATE)

    #evaluate the model
    model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_validation, Y_validation))

    #Test the model

    test_error, test_accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Error: {test_error}, test accuracy: {test_accuracy}")
    #save the model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()
