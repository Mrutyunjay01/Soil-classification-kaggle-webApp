from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, Dense, Flatten


def create_model(inputShape):
    model = Sequential(name="SoilNet")
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (4, 4), padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (4, 4), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (4, 4), padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(128, (4, 4), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    return model


if __name__ == '__main__':
    create_model((150, 150, 3))
