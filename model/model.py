from keras.layers import Input, Conv1D, Conv2D, Flatten, Dense
from keras.layers import TimeDistributed, Multiply
from keras.models import Model


def model_0(input_shape=(250, 54, 96, 1)):
    # don't change since 10:44 9-Aug-2018
    img_input = Input(shape=input_shape)
    of_ave_input = Input(shape=input_shape)
    of_std_input = Input(shape=input_shape)

    img_conv1 = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(img_input)
    ave_conv1 = TimeDistributed(Conv2D(16, (3, 3), activation='sigmoid', padding='same'))(of_ave_input)
    std_conv1 = TimeDistributed(Conv2D(16, (3, 3), activation='sigmoid', padding='same'))(of_std_input)

    multiply = Multiply()([img_conv1, ave_conv1, std_conv1])

    conv2 = TimeDistributed(Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu', use_bias=False))(multiply)
    conv3 = TimeDistributed(Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu', use_bias=False))(conv2)
    conv4 = TimeDistributed(Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu', use_bias=False))(conv3)

    flatten = TimeDistributed(Flatten())(conv4)
    dense1 = TimeDistributed(Dense(1, activation='linear'))(flatten)

    x = Conv1D(16, 17, strides=5, padding='same', activation='relu', use_bias=False)(dense1)
    x = Conv1D(16, 7, strides=5, padding='same', activation='relu', use_bias=False)(x)

    flatten1 = Flatten()(x)
    gap = Dense(1, activation='linear')(flatten1)

    model = Model([img_input, of_ave_input, of_std_input], gap)

    return model