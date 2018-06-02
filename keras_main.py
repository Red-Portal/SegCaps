
import tensorflow as tf
import numpy as np
import keras
import random
import imageio
import os
from custom_losses import *
from keras_model import CapsNetR3

def onehot(output):
    return tf.cast(output > 0.5, dtype=tf.float32)

def load_data(data_path, label_path):
    data = []
    label = []
    for i in os.listdir(data_path):
        data.append(imageio.imread(os.path.join(data_path, i)))
    for i in os.listdir(label_path):
        label.append(imageio.imread(os.path.join(label_path, i)))
    return data, label

def main():
    lr = 0.0001
    report_step = 50
    validation_step = 1000
    total_iteration = 10000
    batch_size = 1

    data, label = load_data("./dataset/imgs", "./dataset/masks")
    print("data: ", len(data), " shape: ", data[0].shape)
    
    data = np.expand_dims(data, -1)
    label = np.expand_dims(label, -1)
    shape = data[0].shape

    input_layer = keras.layers.Input(shape=[shape[0], shape[1], 1])
    model = CapsNetR3(input_layer)
    model = keras.models.Model(inputs=input_layer, outputs=model)
    opt = keras.optimizers.Nadam(lr=lr)
    loss = "binary_crossentropy"#margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    model.fit(data, label,
              validation_split=0.2,
              #validation_step=validation_step,
              #report_step=report_step,
              batch_size=batch_size,
              shuffle=True,
              epochs=5)
    
    # masks = []
    # imags = []
    # for valid_data, valid_label in data_iter.validation():
    #     masks.append(sess.run(op_out, feed_dict={x_in: valid_data}))
    #     imags.append(valid_data)
    # masks = np.concatenate(masks, axis=0)
    # imags = np.concatenate(imags, axis=0)

    # np.save("masks.npy", masks)
    # np.save("imags.npy", imags)

if __name__ == "__main__":
    main()
