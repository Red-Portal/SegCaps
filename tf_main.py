
import tensorflow as tf
import numpy as np
import random
import imageio
import os
from capsnet import CapsNetR3

def soft_jaccard(output, target, axis=(1, 2, 3), smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    l = tf.reduce_sum(output * output, axis=axis)
    r = tf.reduce_sum(target * target, axis=axis)
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice

def onehot(output):
    return tf.cast(output > 0.5, dtype=tf.float32)

def margin_loss(labels, raw_logits, margin=0.4, downweight=0.5, pos_weight=1.0):
    logits = raw_logits - 0.5
    positive_cost = pos_weight * labels \
        * tf.cast(tf.less(logits, margin), tf.float32) * tf.pow(logits - margin, 2)

    negative_cost = (1 - labels) * tf.cast(
        tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
    return 0.5 * positive_cost + downweight * 0.5 * negative_cost

def binary_cross_entropy(labels, logits, smooth=1e-5):
    return tf.reduce_sum(tf.cast(logits > 1.0, dtype=tf.float32))

    positive = labels * tf.log(logits + smooth)
    negative = (1 - labels) * tf.log(1 - logits + smooth)
    return -tf.reduce_mean(positive + negative)

def hard_jaccard(output, target, axis=(1, 2, 3), smooth=1e-5):
    pre = tf.cast(output > 0.5, dtype=tf.float32)
    truth = tf.cast(target > 0.5, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou)
    return iou 

class data_manager:
    class valid_iter:
        def __init__(self, batch_size, valid_data, valid_label):
            self.batch_size = batch_size
            self.valid_data = valid_data
            self.valid_label = valid_label
            self.idx = 0
            return

        def __iter__(self):
            return self

        def __next__(self):
            if self.idx == -1:
                self.idx = 0
                raise StopIteration
            
            begin = self.idx
            end = begin + self.batch_size
            if end > len(self.valid_data):
                end = len(self.valid_data)
                self.idx = -1
            else:
                self.idx = end
            data = self.valid_data[begin:end]
            label = self.valid_label[begin:end]
            return data, label

    def __init__(self, batch_size, train_data, train_label, valid_data, valid_label):
        assert len(train_data) == len(train_label)
        self.batch_size = batch_size
        self.train_data = train_data
        self.train_label = train_label
        self.valid = self.valid_iter(batch_size, valid_data, valid_label)
        self.reset_queue()
        return

    def validation(self):
        return self.valid_data, self.valid_label

    def reset_queue(self):
        self.indices = list(range(len(self.train_data)))
        random.shuffle(self.indices)
        return

    def validation(self):
        return self.valid

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.indices) < self.batch_size:
            self.reset_queue()
        indices = [self.indices.pop() for i in range(self.batch_size)]
        data = np.array([self.train_data[i] for i in indices])
        label = np.array([self.train_label[i] for i in indices])
        return data, label

def load_data(data_path, label_path):
    data = []
    label = []
    for i in os.listdir(data_path):
        data.append(imageio.imread(os.path.join(data_path, i)))
    for i in os.listdir(data_path):
        label.append(imageio.imread(os.path.join(data_path, i)))
    return data, label

def main():
    lr = 0.01
    report_step = 10
    validation_step = 100
    total_iteration = 1000
    batch_size = 16

    data, label = load_data("./dataset/imgs", "./dataset/masks")
    print("data: ", len(data), " shape: ", data[0].shape)
    data_iter = data_manager(batch_size, data[500:], label[500:], data[:500], label[:500])
    shape = data[0].shape
    x_in = tf.placeholder(tf.float32, [None, shape[0], shape[1]])
    y_in = tf.placeholder(tf.float32, [None, shape[0], shape[1]])

    x = tf.expand_dims(x_in , axis=-1)
    y = tf.expand_dims(y_in , axis=-1)
    model = CapsNetR3(x)
    op_loss = binary_cross_entropy(y, model) #margin_loss(y, model)
    op_accu = soft_jaccard(y, model)
    optimizer = tf.contrib.opt.NadamOptimizer(lr)
    op_train = optimizer.minimize(op_loss)
    op_out = onehot(model)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for idx, batch in enumerate(data_iter):
            data, label = batch
            if idx == total_iteration:
                break

            if idx % validation_step == 0:
                stats = []
                for valid_data, valid_label in data_iter.validation():
                    loss, accu = sess.run([op_loss, op_accu],
                                          feed_dict={x_in: valid_data, y_in: valid_label})
                    stats.append([np.mean(loss), np.mean(accu)])
                loss, accu = np.mean(stats, axis=0)
                print("validation loss: ", loss, " accu: ", accu)

            if idx == 0 or idx % report_step == 0:
                loss, accu, _ = sess.run([op_loss, op_accu, op_train],
                                         feed_dict={x_in: data, y_in: label})
                print("step: ", idx, " loss: ", loss, " accu: ", accu)
            else:
                sess.run(op_train, feed_dict={x_in: data, y_in: label})

        masks = []
        imags = []
        for valid_data, valid_label in data_iter.validation():
            masks.append(sess.run(op_out, feed_dict={x_in: valid_data}))
            imags.append(valid_data)
        masks = np.concatenate(masks, axis=0)
        imags = np.concatenate(imags, axis=0)

        np.save("masks.npy", masks)
        np.save("imags.npy", imags)

if __name__ == "__main__":
    main()
