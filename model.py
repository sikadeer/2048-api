from keras.layers import concatenate
from keras.layers import Conv2D, Input, Flatten, BatchNormalization, Activation, LeakyReLU, MaxPooling2D, Dense, Dropout
from keras.models import Model


def build():
    inputs = Input((4, 4, 16))
    FILTERS = 128
    conv_a = Conv2D(filters=FILTERS, kernel_size=(2, 1), kernel_initialzer='he_uniform')(inputs)
    conv_b = Conv2D(filters=FILTERS, kernel_size=(1, 2), kernel_initialzer='he_uniform')(inputs)
    conv_aa = Conv2D(filters=FILTERS, kernel_size=(2, 1), kernel_initialzer='he_uniform')(conv_a)
    conv_ab = Conv2D(filters=FILTERS, kernel_size=(1, 2), kernel_initialzer='he_uniform')(conv_a)
    conv_ba = Conv2D(filters=FILTERS, kernel_size=(2, 1), kernel_initialzer='he_uniform')(conv_b)
    conv_bb = Conv2D(filters=FILTERS, kernel_size=(1, 2), kernel_initialzer='he_uniform')(conv_b)

    conv_a = LeakyReLU(alpha=0.3)(conv_a)
    conv_b = LeakyReLU(alpha=0.3)(conv_b)
    conv_aa = LeakyReLU(alpha=0.3)(conv_aa)
    conv_ab = LeakyReLU(alpha=0.3)(conv_ab)
    conv_ba = LeakyReLU(alpha=0.3)(conv_ba)
    conv_bb = LeakyReLU(alpha=0.3)(conv_bb)

    conv1 = MaxPooling2D()(conv_a)
    conv2 = MaxPooling2D()(conv_b)
    conv11 = MaxPooling2D()(conv_aa)
    conv12 = MaxPooling2D()(conv_ab)
    conv21 = MaxPooling2D()(conv_ba)
    conv22 = MaxPooling2D()(conv_bb)

    hidden = concatenate([Flatten()(conv1),
                          Flatten()(conv2),
                          Flatten()(conv11),
                          Flatten()(conv12),
                          Flatten()(conv21),
                          Flatten()(conv22)])

    x = BatchNormalization()(hidden)
    x = Activation('relu')(x)
    #
    for width in [512, 256]:
        x = Dense(width, kernel_initializer="he_uniform")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Dropout(0.3)(x)
    outputs = Dense(4, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.summary()
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model
