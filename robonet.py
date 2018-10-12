import keras
#import pydot
import pydotplus as pydot

from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, MaxPooling2D,Flatten,Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Activation
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

input_1 = Input(shape=(32, 32, 3), name="head")

x1 = Conv2D(16, (3, 3), padding='same', name='2-1')(input_1)
x2 = Conv2D(16, (3, 3), padding='same', name='2-2')(input_1)
x3 = Conv2D(16, (3, 3), padding='same', name='2-3')(input_1)

x4 = Conv2D(10, (3, 3), padding='same', name='3-1')(x1)
x5 = Concatenate(axis=3, name="3-2")([x1,x2])
x6 = Concatenate(axis=3, name="3-3")([x2,x3])
x7 = Conv2D(10, (3, 3), padding='same', name='3-4')(x3)

out1=GlobalAveragePooling2D(name="4-1")(x4)
out1 = Activation('softmax', name='r_hand')(out1)

x10 = Concatenate(axis=3, name="4-2")([x5,x6])

out4=GlobalAveragePooling2D(name="4-3")(x7)
out4 = Activation('softmax', name='l_hand')(out4)

x11 = MaxPooling2D(pool_size=(2, 2),name="5-1")(x10)

out2 = Flatten(name="6-1")(x11)
out2 = Dense(10, activation='relu',name="7-1")(out2)
out2 = Activation('softmax',name="r_foot")(out2)

out3 = Flatten(name="6-2")(x11)
out3 = Dense(10, activation='relu',name="7-2")(out3)
out3 = Activation('softmax',name="l_foot")(out3)

model = Model(inputs=[input_1],
                outputs=[out1,out2,out3,out4])

#モデルの保存
plot_model(model, to_file='robonet.png')

#モデルの図示
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(model).create(prog='dot', format='svg'))

# CIFAR-10データをロード
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 0-1に変換
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

# クラスラベル（0-9）をone-hotエンコーディング形式に変換
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy','categorical_crossentropy'],
              optimizer=Adam(),
              metrics=['accuracy'])

early_stopping = EarlyStopping(patience=3, verbose=1)

history = model.fit(X_train,
                    [Y_train, Y_train, Y_train, Y_train],
                    batch_size=128,
                    nb_epoch=100,
                    verbose=1,
                    validation_split=0.2,
                    callbacks=[early_stopping])
