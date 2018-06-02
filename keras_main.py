
import tensorflow as tf
import numpy as np
import keras
import random
import imageio
import os
from custom_losses import *
from keras_model import CapsNetR3

def hard_jaccard(y_true, y_pred, smooth=1e-5):
    # intersection = keras.backend.sum(keras.backend.abs(y_true * y_pred), axis=-1)
    # sum_ = keras.backend.sum(keras.backend.abs(y_true) + keras.backend.abs(y_pred), axis=-1)
    # jac = (intersection + smooth) / (sum_ - intersection + smooth)
    # return (1 - jac)

    y_true = keras.backend.batch_flatten(y_true)
    y_true = keras.backend.cast(keras.backend.greater(y_true, 0.5), "float32")

    y_pred = keras.backend.batch_flatten(y_pred)
    y_pred = keras.backend.cast(keras.backend.greater(y_pred, 0.5), "float32")

    inter = y_true * y_pred
    inter = keras.backend.sum(inter, axis=-1)

    total = keras.backend.sum(y_true, axis=-1) + keras.backend.sum(y_pred, axis=-1)
    union = total - inter

    jac = (inter + smooth) / (union + smooth)
    return keras.backend.mean(jac)

def soft_jaccard(y_true, y_pred, smooth=100):
    intersection = keras.backend.sum(keras.backend.abs(y_true * y_pred), axis=-1)
    sum_ = keras.backend.sum(keras.backend.abs(y_true) + keras.backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

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
    validation_split = 0.2
    epochs = 1
    batch_size = 1

    data, label = load_data("./dataset/imgs", "./dataset/masks")
    print("data: ", len(data), " shape: ", data[0].shape)

    test_data, test_label = load_data("./dataset/test_imgs", "./dataset/test_masks")
    print("test data: ", len(data), " shape: ", data[0].shape)
    
    data = np.expand_dims(np.array(data), -1)
    label = np.expand_dims(np.array(label), -1)
    shape = data[0].shape

    test_data = np.expand_dims(np.array(test_data), -1)
    test_label = np.expand_dims(np.array(test_label), -1)

    input_layer = keras.layers.Input(shape=[shape[0], shape[1], 1])
    model = CapsNetR3(input_layer)
    model = keras.models.Model(inputs=input_layer, outputs=model)
    opt = keras.optimizers.Nadam(lr=lr)
    loss = "binary_crossentropy"#margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0)
    model.compile(optimizer=opt, loss=soft_jaccard, metrics=[hard_jaccard])
    model.fit(data, label,
              validation_split=validation_split,
              #validation_step=validation_step,
              #report_step=report_step,
              batch_size=batch_size,
              shuffle=True,
              epochs=epochs)
    print("-- saving model")
    model.save("segcaps.h5")
    print("-- saving model - success")

    metrics =  model.evaluate(test_data, test_label, batch_size=batch_size, verbose=1)
    print("final metrics: ", metrics)

    masks = []
    imags = []
    for i in range(test_data.shape[0]):
        masks.append(model.predict(test_data[i:i+1,:,:,:]))
        imags.append(test_data[i:i+1,:,:,:])
    masks = np.concatenate(masks, axis=0)
    imags = np.concatenate(imags, axis=0)

    masks = (masks > 0.5)
    inter = (masks * test_label > 0.5)
    union = (masks + test_label > 0.5)

    np.save("inter.npy", inter)
    np.save("union.npy", union)

    np.save("masks.npy", masks)
    np.save("imags.npy", imags)
    print("saved images: ", imags.shape, " masks: ", masks.shape)

if __name__ == "__main__":
    main()
