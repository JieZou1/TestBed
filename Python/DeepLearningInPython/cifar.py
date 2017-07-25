import keras
from keras import backend
from keras.callbacks import ModelCheckpoint
from keras.datasets.cifar import load_batch
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import get_file, to_categorical
import numpy as np
import os


num_classes = 10
data_augmentation = False
batch_size = 32
epochs = 200


def load_cifar10():
    # download and extract data
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin, untar=True, cache_dir='Z:\\', cache_subdir="datasets")

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    # load train data
    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i-1)*10000:i*10000, :, :, :] = data
        y_train[(i-1)*10000:i*10000] = labels

    # load test data
    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if backend.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def normalize_data(x_train, y_train, x_test, y_test):
    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)


def cifar10_train(model_file=None):

    # Load and normalize the data
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    (x_train, y_train), (x_test, y_test) = normalize_data(x_train, y_train, x_test, y_test)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Create or Load model
    if model_file is None:
        print('Create a new model.')
        # create model
        model = create_cnn_model(num_classes, x_train.shape[1:])
        # Train the model using RMSprop
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    else:
        print('Load existing model', model_file)
        model = load_model(model_file)

    model.summary()

    # Training
    model = fit_model(x_train, y_train, x_test, y_test, model)

    # Evaluate
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    pass


def create_cnn_model(num_classes, input_shape):
    model = Sequential()
    # "valid" means there is no padding around input or feature map,
    # "same" means there are some padding around input or feature map,
    # making the output feature map's size same as the input's
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    # model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    # model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # model.add(Activation('softmax'))

    return model


def fit_model(x_train, y_train, x_test, y_test, model):

    # Checkpointing is setup to save the network weights only when there is an improvement in classification accuracy
    # on the validation dataset (monitor=’val_acc’ and mode=’max’).
    filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks_list,
                  verbose=0)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            callbacks=callbacks_list,
                            verbose=0)
    return model


def cifar10_test():
    num_classes = 10
    model = load_model('cifar_models/augmented/weights-improvement-62-0.80.hdf5')
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = to_categorical(y_test, num_classes)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    pass


if __name__ == "__main__":
    cifar10_train()
    # cifar10_test()
    pass
