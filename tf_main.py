
import tensorflow as tf
import random
import imageio
from capsnet import CapsNetR3

def soft_jaccard(output, target, axis=(1, 2, 3), smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    l = tf.reduce_sum(output * output, axis=axis)
    r = tf.reduce_sum(target * target, axis=axis)
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice

def hard_jaccard(output, target, axis=(1, 2, 3), smooth=1e-5):
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou)
    return iou 

class data_manager:
    def __init__(self, train_data, train_label,
                 valid_data, valid_label):
        assert len(data) == len(label)
        self.train_data = train_data
        self.train_label = train_label
        self.valid_data = valid_data
        self.valid_label = valid_label
        self.reset_queue()
        return

    def validation(self):
        return self.valid_data, self.valid_label

    def reset_queue(self):
        self.indices = list(range(len(data)))
        random.shuffle(self.indices)
        return

    def __iter__(self):
        return self

    def __next__(self, batch_size):
        if len(self.indices) < batch_size:
            self.reset_queue()
        indices = [self.indices.pop() for i in range(batch_size)]
        data = np.array(self.train_data[indices])
        label = np.array(self.train_label[indices])
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
    lr = 0.0001
    report_step = 10
    validation_step = 100
    total_iteration = 1000

    data, label = load_data("data", "label")
    data_iter = data_manager(data, label)
    x_in = tf.placeholder(tf.float32, [None, 512, 256])
    y_in = tf.placeholder(tf.float32, [None, 512, 256])

    x = tf.expand_dims(x_in , axis=-1)
    model = CapsNetR3(x)
    op_loss = soft_jaccard(x, model)
    op_accu = hard_jaccard(x, model)
    optimizer = tf.train.NadamOptimizer(lr)
    op_train = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for idx, data, label in enumerate(data_iter):
            if idx == total_iteration:
                break

            if idx % validation_step == 0:
                valid_data, valid_label = data_iter.validation()
                loss, accur = sess.run([op_loss, op_accu],
                                        feed_dict={x_in: valid_data, y_in: valid_label})
                print("validation loss: ", loss, " accu: ", accu)

            if idx == 0 or idx % report_step == 0:
                loss, accur, _ = sess.run([op_loss, op_accu, op_train],
                                          feed_dict={x_in: data, y_in: label})
                print("step: ", idx, " loss: ", loss, " accu: ", accu)
            else:
                sess.run(op_train, feed_dict={x_in: data, y_in: label})
    return

if __name__ == "__main__":
    main()
