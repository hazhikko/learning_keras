from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img, list_pictures
import numpy as np
import os, sys, pprint

# 学習済みmodel
MODEL_DATA = './source/deep-learning-with-keras-ja/ch01/model/keras_MINST_V6_1_model.h5'
# テスト画像ディレクトリ
IMAGE_DIRS = [
    './source/deep-learning-with-keras-ja/ch01/color_white',
    './source/deep-learning-with-keras-ja/ch01/grayscale_black'
]
# IMAGE_DIR1 = './source/deep-learning-with-keras-ja/ch01/color_white'
# IMAGE_DIR2 = './source/deep-learning-with-keras-ja/ch01/grayscale_black'
# カテゴリー
CATEGORIES = ['0','1','2','3','4','5','6','7','8','9']

def normalization(data):
    data = data.astype('float32')
    data = data / 255.0
    return data

if __name__ == '__main__':
    X = []
    file_path_list = []

    for dirs in IMAGE_DIRS:
        for picture in list_pictures(dirs):
            file_path_list.append(picture)
            img = load_img(picture, grayscale=True, target_size=(28, 28))
            img = img_to_array(img)
            X.append(img)

    X = np.asarray(X)
    X = X.reshape(len(X), 784)
    X = normalization(X)

    model = load_model(MODEL_DATA)

    predict = model.predict_proba(X)
    for i, pre in enumerate(predict):
        idx = np.argmax(pre)
        rate = pre[idx] * 100
        print(file_path_list[i] + ':' + str(CATEGORIES[idx]) + ':' + str(round(rate, 5)))
