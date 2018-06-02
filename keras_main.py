
import tensorflow as tf
import numpy as np
import keras
import random
import imageio
import os
from custom_losses import *
from keras_model import CapsNetR3

def jaccard_distance(y_true, y_pred, smooth=1e-5):
    """Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # References
    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
    https://en.wikipedia.org/wiki/Jaccard_index
    """
    y_true = keras.backend.batch_flatten(y_true)

    y_pred = keras.backend.batch_flatten(y_pred)
    y_pred = keras.backend.cast(keras.backend.greater(y_pred, 0.5), "float32")

    inter = y_true * y_pred

    union = y_true + y_pred
    union = keras.backend.cast(keras.backend.greater(union, 0.5), "float32")

    jac = (inter + smooth) / (union + smooth)
    return keras.backend.mean(jac)

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
    batch_size = 4

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
    model.compile(optimizer=opt, loss=loss, metrics=[jaccard_distance])
    model.fit(data, label,
              validation_split=validation_split,
              #validation_step=validation_step,
              #report_step=report_step,
              batch_size=batch_size,
              shuffle=True,
              epochs=epochs)
    model.save("segcaps.h5")

    metrics =  model.evaluate(test_data, test_label, batch_size=batch_size, verbose=1)
    print("final metrics: ", metrics)

    masks = []
    imags = []
    for idx in range(test_data.shape[0]):
        masks.append(model.predict(test_data[i:i+1,:,:,:]))
        imags.append(test_data[i:i+1,:,:,:])
    masks = np.concatenate(masks, axis=0)
    imags = np.concatenate(imags, axis=0)

    np.save("masks.npy", masks)
    np.save("imags.npy", imags)
    print("saved images: ", imags.shape, " masks: ", masks.shape)

if __name__ == "__main__":
    main()
