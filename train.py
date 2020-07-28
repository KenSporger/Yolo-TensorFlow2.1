import tensorflow as tf
from model import *
from voc_data import *
from calculation import *
from tensorflow.keras import optimizers

input_shape = (224, 224, 3)
output_shape = (7,7,30)

if __name__ == '__main__':
    # 初始化GPU，内存分配用多少分配多少
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model = yolov1_model(input_shape, output_shape)
    model.compile(
        loss=calcu_loss,
        optimizer=optimizers.Adam()
    )
    model.summary()
    train_sequence = SequenceData("./train.txt", (224, 224), (7, 7), 32, 20)
    val_sequence = SequenceData("./val.txt", (224, 224), (7, 7), 32, 20)
    model.fit(
        train_sequence,
        steps_per_epoch=train_sequence.get_epochs(),
        validation_data=val_sequence,
        validation_steps=val_sequence.get_epochs(),
        epochs=100,
        workers=6,#最大线程数
    )
    model.save_weights("model_data/yolov2_trained.h5")
    print("over....")
