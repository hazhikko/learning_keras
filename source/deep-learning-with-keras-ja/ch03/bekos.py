import os, glob, time, datetime
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten, Dropout
from keras.layers.core import Dense
from keras.datasets import cifar10
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def network(input_shape, num_classes):
    model = Sequential()

    # extract image features by convolution and max pooling layers
    model.add(Conv2D(
        32, kernel_size=3, padding='same',
        input_shape=input_shape, activation='relu'
        ))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # classify the class by fully-connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


class BEKOSDataset():

    def __init__(self):
        self.image_shape = (32, 32, 3)
        self.num_classes = 5

    def get_batch(self):
        folder = ['bekosus','chimairabeko','kerbecos','minotaubeko','unibecorn']
        image_size = 32

        X = []
        Y = []
        for index, name in enumerate(folder):
            dir = './bekos/data/' + name
            files = glob.glob(dir + '/*.jpg')
            for file in files:
                image = Image.open(file)
                image = image.convert('RGB')
                image = image.resize((image_size, image_size))
                data = np.asarray(image)
                X.append(data)
                Y.append(index)

        X = np.array(X)
        Y = np.array(Y)

        #画像データを0から1の範囲に変換
        X = X.astype('float32')
        X = X / 255.0

        # 正解ラベルの形式を変換
        Y = np_utils.to_categorical(Y, len(folder))

        # 学習用データとテストデータ
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

        return x_train, y_train, x_test, y_test

    def preprocess(self, data, label_data=False):
        if label_data:
            # convert class vectors to binary class matrices
            data = keras.utils.to_categorical(data, self.num_classes)
        else:
            data = data.astype('float32')
            data /= 255  # convert the value to 0~1 scale
            shape = (data.shape[0],) + self.image_shape  # add dataset length
            data = data.reshape(shape)

        return data


class Trainer():

    def __init__(self, model, loss, optimizer):
        self._target = model
        self._target.compile(
            loss=loss, optimizer=optimizer, metrics=['accuracy']
            )
        self.verbose = 1
        logdir = 'logdir_bekos'
        self.log_dir = os.path.join(os.path.dirname(__file__), logdir)
        self.model_file_name = 'model_file.hdf5'

    def train(self, x_train, y_train, batch_size, epochs, validation_split):
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)  # remove previous execution
        os.mkdir(self.log_dir)

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (0~180)
            width_shift_range=0.1,  # randomly shift images horizontally
            height_shift_range=0.1,  # randomly shift images vertically
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities for normalization (mean, std etc)
        datagen.fit(x_train)

        # split for validation data
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        validation_size = int(x_train.shape[0] * validation_split)
        x_train, x_valid = \
            x_train[indices[:-validation_size], :], \
            x_train[indices[-validation_size:], :]
        y_train, y_valid = \
            y_train[indices[:-validation_size], :], \
            y_train[indices[-validation_size:], :]

        model_path = os.path.join(self.log_dir, self.model_file_name)
        self._target.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=x_train.shape[0] // batch_size,
            epochs=epochs,
            validation_data=(x_valid, y_valid),
            callbacks=[
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(model_path, save_best_only=True)
            ],
            verbose=self.verbose,
            workers=5
        )


dataset = BEKOSDataset()

# make model
model = network(dataset.image_shape, dataset.num_classes)

# 学習の開始時刻を取得
start_time = time.time()

# train the model
x_train, y_train, x_test, y_test = dataset.get_batch()
trainer = Trainer(model, loss='categorical_crossentropy', optimizer=RMSprop())
trainer.train(
    x_train, y_train, batch_size=32, epochs=50, validation_split=0.2
    )

# 学習と検証の終了時刻を取得
end_time = time.time()

# show result
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 学習処理の時間を表示
elapsed_time = end_time - start_time
td = datetime.timedelta(seconds=elapsed_time)
print('学習処理時間：{0}'.format(str(td)))