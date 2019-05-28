from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from make_tensorboard import make_tensorboard
import matplotlib.pyplot as plt
import time, datetime

# randomで生成する値を固定する
np.random.seed(1671)

# 各パラメータ設定
NB_EPOCH = 250          # 学習を繰り返す回数
BATCH_SIZE = 128        # 学習の精度と速度に関わる 参考：https://www.st-hakky-blog.com/entry/2017/11/16/161805
VERBOSE = 1             # 進行状況の表示モード 1=プログレスバーで表示
NB_CLASSES = 10         # OHEするときのクラス数(0～9の数値なので10)
OPTIMIZER = SGD()       # 学習時に損失値をできるだけ小さくする=精度を上げるための手法 SGD=確率的勾配降下法
N_HIDDEN = 128          # 隠れ層
VALIDATION_SPLIT = 0.2  # 使用するデータのうち、検証に使用するデータの割合 0.2=20%
DROPOUT = 0.3           # ドロップアウト率 30%がランダムに間引かれ、値がゼロになる

# 画像データの読み込み
# X_train   学習用の画像データ
# X_test    学習用の正解データ
# Y_train   検証用の画像データ
# Y_test    検証用の正解データ
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 画像を1次元に変換する
# 784要素の1次元配列 x 60,000になる
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)

# データ型をfloatにする
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# pixelの情報は黒～白(0～255)の数値
# 0～1の範囲の数値にするため、255で割る
X_train /= 255
X_test /= 255

# 学習用データの数
# 60000 train samples
# 10000 test samples
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 0～9の数値をOHEの10要素1次元配列にする
# 0：[1,0,0,0,0,0,0,0,0,0]
# 1：[0,1,0,0,0,0,0,0,0,0]
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

# Sequentialモデルを使用する
model = Sequential()

# layer：Dense
# 活性化関数：relu,softmax
# 隠れ層N_HIDDENを追加
# ドロップアウトを追加
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

# モデルの要約を出力 summary：utils.print_summaryのエイリアス
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_1 (Dense)              (None, 128)               100480
# _________________________________________________________________
# activation_1 (Activation)    (None, 128)               0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 128)               0
# _________________________________________________________________
# dense_2 (Dense)              (None, 128)               16512
# _________________________________________________________________
# activation_2 (Activation)    (None, 128)               0
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 128)               0
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                1290
# _________________________________________________________________
# activation_3 (Activation)    (None, 10)                0
# =================================================================
# Total params: 118,282
# Trainable params: 118,282
# Non-trainable params: 0
# _________________________________________________________________
model.summary()

# 学習モデルの構築
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# 学習の履歴を可視化するTensorBoardの結果を受け取るcallbackの設定
callbacks = [make_tensorboard(set_dir_name='keras_MINST_V3_2')]

# 学習の開始時刻を取得
start_time = time.time()

# 学習の実行
# 学習の直前と毎epochの終了時にTensorBoardが呼び出され、結果が出力される
# Epoch 1/200
# 48000/48000 [==============================] - 1s 25us/step - loss: 1.3633 - acc: 0.6796 - val_loss: 0.8904 - val_acc: 0.8246
# Epoch 2/200
# 48000/48000 [==============================] - 1s 21us/step - loss: 0.7913 - acc: 0.8272 - val_loss: 0.6572 - val_acc: 0.8546
# ・・・
# Epoch 200/200
# 48000/48000 [==============================] - 1s 21us/step - loss: 0.2761 - acc: 0.9230 - val_loss: 0.2756 - val_acc: 0.9241
# 10000/10000 [==============================] - 0s 27us/step
result = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, callbacks=callbacks, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# 学習結果を検証用データを使って評価する
# Test score: 0.277385850328    値が小さいほど正しい結果を出せている
# Test accuracy: 0.9227         検証の正解率　92%
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])

# 学習と検証の終了時刻を取得
end_time = time.time()

# 学習処理の時間を表示
elapsed_time = end_time - start_time
td = datetime.timedelta(seconds=elapsed_time)
print('学習処理時間：{0}'.format(str(td)))

# 学習経過をmatplotlibでグラフにする
# X軸：Epoch数
# Y軸：正解率
plt.plot(range(1, NB_EPOCH + 1), result.history['acc'], label='traning')
plt.plot(range(1, NB_EPOCH + 1), result.history['val_acc'], label='validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()