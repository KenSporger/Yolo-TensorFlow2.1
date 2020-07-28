import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, MaxPool2D, \
                                Input, Lambda, concatenate, Flatten, Dense, Dropout, Reshape
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def yolov1_conv2D(x, n_filter, ksize, strides=1, pool=False):
    x = Conv2D(
        filters=n_filter,
        kernel_size=ksize,
        strides=strides,
        padding='same',
        kernel_initializer='TruncatedNormal'
    )(x)
    x = LeakyReLU(alpha=0.1)(x)
    if (pool):
        x = MaxPool2D(strides=2, padding='same')(x)
    return x

def yolov1_model(input_shape, output_shape):
    input_layer = Input(shape=input_shape)
    x = yolov1_conv2D(input_layer, 64, 7, strides=2, pool=True)

    x = yolov1_conv2D(x, 192, 3, pool=True)

    x = yolov1_conv2D(x, 128, 1)
    x = yolov1_conv2D(x, 256, 3)
    x = yolov1_conv2D(x, 256, 1)
    x = yolov1_conv2D(x, 512, 3, pool=True)

    x = yolov1_conv2D(x, 256, 1)
    x = yolov1_conv2D(x, 512, 3)
    x = yolov1_conv2D(x, 256, 1)
    x = yolov1_conv2D(x, 512, 3)
    x = yolov1_conv2D(x, 256, 1)
    x = yolov1_conv2D(x, 512, 3)
    x = yolov1_conv2D(x, 256, 1)
    x = yolov1_conv2D(x, 512, 3)
    x = yolov1_conv2D(x, 512, 1)
    x = yolov1_conv2D(x, 1024, 3, pool=True)

    x = yolov1_conv2D(x, 512, 1)
    x = yolov1_conv2D(x, 1024, 3)
    x = yolov1_conv2D(x, 512, 1)
    x = yolov1_conv2D(x, 1024, 3)
    x = yolov1_conv2D(x, 1024, 3)
    x = yolov1_conv2D(x, 1024, 3, strides=2)

    x = yolov1_conv2D(x, 1024, 3)
    x = yolov1_conv2D(x, 1024, 3)

    x = Flatten()(x)
    x = Dense(4096, kernel_initializer='TruncatedNormal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(2940, kernel_initializer='TruncatedNormal')(x)
    output_layer = Reshape(output_shape)(x)

    model = tf.keras.Model(
        inputs = input_layer,
        outputs = output_layer,
        name="Yolov1"
    )

    return model


if __name__ == '__main__':
    model = yolov1_model(input_shape=(448,448,3), output_shape=(7,7,30))
    model.summary()