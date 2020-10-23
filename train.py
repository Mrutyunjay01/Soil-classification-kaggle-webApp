from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from src.model import create_model

TRAIN_PATH = "input/Train/"
VAL_PATH = "input/Test"
WEIGHT_PATH = "weights/"
if __name__ == '__main__':
    train_gen = ImageDataGenerator(rescale=1 / 255.,
                                   rotation_range=45,
                                   zoom_range=.4,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.15,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode="nearest")

    train_data = train_gen.flow_from_directory(TRAIN_PATH,
                                               target_size=(150, 150),
                                               batch_size=64)

    val_gen = ImageDataGenerator(rescale=1 / 255.0)

    val_data = val_gen.flow_from_directory(VAL_PATH,
                                           target_size=(150, 150),
                                           batch_size=64)

    model = create_model(inputShape=(150, 150, 3))
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=20)

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()
    plt.savefig('LossVal_loss')

    # plot the accuracy
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.show()
    plt.savefig('AccVal_acc')

    print("\nSaving model...")
    model.save(WEIGHT_PATH + "model_v1.h5")
    pass
