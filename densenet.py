import numpy as np
import tensorflow as tf
class DCNN:
    epoch=100
    batch_size=100
    def __init__(self):
        return
    def set_epoch(self,value):
        self.epoch=value
    def set_batch_size(self,value):
        self.batch_size=value
    def pre_process_image(self,image, training):
        # This function takes a single image as input,
        # and a boolean whether to build the training or testing graph.
        img_size_cropped = 32
        if training==True:
            # For training, add the following to the TensorFlow graph.

            # Randomly crop the input image.
            image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, 3])

            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)

            # Randomly adjust hue, contrast and saturation.
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

            # Some of these functions may overflow and result in pixel
            # values beyond the [0, 1] range. It is unclear from the
            # documentation of TensorFlow 0.10.0rc0 whether this is
            # intended. A simple solution is to limit the range.

            # Limit the image pixels between [0, 1] in case of overflow.
            image = tf.minimum(image, 1.0)
            image = tf.maximum(image, 0.0)
        else:
            # For training, add the following to the TensorFlow graph.

            # Crop the input image around the centre so it is the same
            # size as images that are randomly cropped during training.
            image = tf.image.resize_image_with_crop_or_pad(image,
                                                           target_height=img_size_cropped,
                                                           target_width=img_size_cropped)

        return image

    def pre_process(self,images, training):
        # Use TensorFlow to loop over all the input images and call
        # the function above which takes a single image as input.
        images = tf.map_fn(lambda image: self.pre_process_image(image,training), images)
        return images

    def run_in_batch_avg(self,session, tensors, batch_placeholders, feed_dict={}):
        res = [0] * len(tensors)
        batch_tensors = [(placeholder, feed_dict[placeholder]) for placeholder in batch_placeholders]
        total_size = len(batch_tensors[0][1])
        batch_count = (total_size + self.batch_size - 1) / self.batch_size
        for batch_idx in xrange(batch_count):
            current_batch_size = None
            for (placeholder, tensor) in batch_tensors:
                batch_tensor = tensor[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                current_batch_size = len(batch_tensor)
                feed_dict[placeholder] = tensor[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
            tmp = session.run(tensors, feed_dict=feed_dict)
            res = [r + t * current_batch_size for (r, t) in zip(res, tmp)]
        return [r / float(total_size) for r in res]

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)
    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
    def conv2d(self,input, in_features, out_features, kernel_size, with_bias=False):
        W = self.weight_variable([kernel_size, kernel_size, in_features, out_features])
        conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
        if with_bias:
            return conv + self.bias_variable([out_features])
        return conv
    def batch_activ_conv(self,current, in_features, out_features, kernel_size, is_training, keep_prob):
        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
        current = tf.nn.relu(current)
        current = self.conv2d(current, in_features, out_features, kernel_size)
        current = tf.nn.dropout(current, keep_prob)
        return current
    def block(self,input, layers, in_features, growth, is_training, keep_prob):
        current = input
        features = in_features
        for idx in xrange(layers):
            tmp = self.batch_activ_conv(current, features, growth, 3, is_training, keep_prob)
            current = tf.concat([current, tmp], 3)
            features += growth
        return current, features
    def avg_pool(self,input, s):
        return tf.nn.avg_pool(input, [1, s, s, 1], [1, s, s, 1], 'VALID')

    def cnn_model(self,images,dim,label_count,layers,is_training,keep_prob):
        current = tf.reshape(images, [-1, dim[0], dim[1], dim[2]])
        #current = self.pre_process(images=current,training=is_training)
        current = self.conv2d(current, 3, 16, 3)
        current, features = self.block(current, layers, 16, 12, is_training, keep_prob)

        current = self.batch_activ_conv(current, features, features, 1, is_training, keep_prob)
        current = self.avg_pool(current, 2)
        current, features = self.block(current, layers, features, 12, is_training, keep_prob)

        current = self.batch_activ_conv(current, features, features, 1, is_training, keep_prob)
        current = self.avg_pool(current, 2)
        current, features = self.block(current, layers, features, 12, is_training, keep_prob)

        current = self.batch_activ_conv(current, features, features, 1, is_training, keep_prob)
        current = self.avg_pool(current, 2)
        current, features = self.block(current, layers, features, 12, is_training, keep_prob)

        current = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
        current = tf.nn.relu(current)
        current = self.avg_pool(current, 4)
        final_dim = features
        current = tf.reshape(current, [-1, final_dim])
        Wfc = self.weight_variable([final_dim, label_count])
        bfc = self.bias_variable([label_count])
        return  tf.nn.softmax(tf.matmul(current, Wfc) + bfc)

    def train(self,data, image_dim, label_count, depth):
        weight_decay = 1e-4
        layers = (depth - 4) / 3
        graph = tf.Graph()
        with graph.as_default():
            xs = tf.placeholder("float", shape=[None, image_dim[0]*image_dim[1]*image_dim[2]])
            ys = tf.placeholder("float", shape=[None, label_count])
            lr = tf.placeholder("float", shape=[])
            keep_prob = tf.placeholder(tf.float32)
            is_training = tf.placeholder("bool", shape=[])
            prediction_probability=self.cnn_model(xs,image_dim,label_count,layers,is_training,keep_prob)
            cross_entropy = -tf.reduce_mean(ys * tf.log(prediction_probability + 1e-12))
            l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)
            correct_prediction = tf.equal(tf.argmax(prediction_probability, 1), tf.argmax(ys, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        with tf.Session(graph=graph) as session:
            learning_rate = 0.1
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            train_data, train_labels = data['train_data'], data['train_labels']
            batch_count = len(train_data) / self.batch_size
            batches_data = np.split(train_data[:batch_count * self.batch_size], batch_count)
            batches_labels = np.split(train_labels[:batch_count * self.batch_size], batch_count)
            print "Batch per epoch: ", batch_count
            for epoch in xrange(1, 1 + self.epoch):
                if epoch == 150: learning_rate = 0.01
                if epoch == 225: learning_rate = 0.001
                for batch_idx in xrange(batch_count):
                    xs_, ys_ = batches_data[batch_idx], batches_labels[batch_idx]
                    batch_res = session.run([train_step, cross_entropy, accuracy],
                                            feed_dict={xs: xs_, ys: ys_, lr: learning_rate, is_training: True,
                                                       keep_prob: 0.8})
                    if batch_idx % 100 == 0: print epoch, batch_idx, batch_res[1:]

                save_path = saver.save(session, 'densenet_%d.ckpt' % epoch)
                test_results = self.run_in_batch_avg(session, [cross_entropy, accuracy], [xs, ys],
                                                feed_dict={xs: data['test_data'], ys: data['test_labels'],
                                                           is_training: False, keep_prob: 1.})
                print epoch, batch_res[1:], test_results


def run():
    from dataset_reader import read_cifar10
    data = read_cifar10('datasets/cifar-10-batches-py/', one_hot=True)
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']
    text_labels = data['text_labels']
    label_count = len(text_labels)
    pi = np.random.permutation(len(train_x))
    train_x, train_y = train_x[pi], train_y[pi]
    print "Train:", np.shape(train_x), np.shape(train_y)
    print "Test:", np.shape(test_x), np.shape(test_y)
    model=DCNN()
    model.set_epoch(300)
    model.set_batch_size(100)
    data = {'train_data': train_x,
            'train_labels': train_y,
            'test_data': test_x,
            'test_labels': test_y}
    model.train(data, [32 , 32 , 3], label_count, 40)
run()
