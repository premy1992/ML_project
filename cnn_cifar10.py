import tensorflow as tf
import numpy as np
import utils

n_classes = 10
batch_size = 128
feature_size = 1024*3

def cnn(train_x, train_y, test_x, test_y):
    def convnet(images):
        images = tf.reshape(images, shape=[-1, 32, 32, 3])

        with tf.variable_scope('conv1') as scope:
            weights = tf.get_variable('weights',
                                      shape=[3, 3, 3, 96],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[96],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)

        # pool1 and norm1
        with tf.variable_scope('pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pooling1')
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            weights = tf.get_variable('weights',
                                      shape=[3, 3, 96, 64],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[64],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name='conv2')

        # pool2 and norm2
        with tf.variable_scope('pooling2_lrn') as scope:
            norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                                   padding='SAME', name='pooling2')

        # local3
        with tf.variable_scope('local3') as scope:
            dim = 16 * 16 * 64
            reshape = tf.reshape(pool2, [-1, dim])
            weights = tf.get_variable('weights',
                                      shape=[dim, 384],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[384],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # local4
        with tf.variable_scope('local4') as scope:
            weights = tf.get_variable('weights',
                                      shape=[384, 192],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[192],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

        # softmax
        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.get_variable('softmax_linear',
                                      shape=[192, 10],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[10],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

        return softmax_linear



    x = tf.placeholder('float', [None, feature_size])
    y = tf.placeholder('float', [None, n_classes])
    prediction = convnet(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # learningRate = tf.train.exponential_decay(learning_rate=0.01,global_step=1,
    #                                         decay_steps=train_x.shape[0],decay_rate=0.02,staircase=True)
    optimizer = tf.train.AdamOptimizer(00.0001).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    epochs = 100
    pred = None
    accu = None
    for epoch in range(epochs):
        epoch_loss = 0
        i = 0
        while i < len(train_x):
            start = i
            end = i + batch_size
            if (end > len(train_x)): end = len(train_x)
            batch_x = np.array(train_x[start:end])
            try:
                batch_x = batch_x.reshape(end - start, feature_size)
            except:
                print("check train_x shape")
            batch_y = np.array(train_y[start:end])
            try:
                batch_y = batch_y.reshape(end - start, n_classes)
            except:
                print("check train_y shape")
            _, c = session.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            epoch_loss += c
            i += batch_size
        if test_x is not None and test_y is not None:
            test_x = test_x.reshape(len(test_x), feature_size)
            test_y = test_y.reshape(len(test_y), n_classes)
            predictio = tf.argmax(prediction, 1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predictio, tf.argmax(y, 1)), "float"))
            pred, accu = session.run([predictio, accuracy], feed_dict={x: test_x[0:500, :], y: test_y[0:500, :]})
            print ('Epoch', epoch + 1, ' completed out of', epochs, 'loss:', epoch_loss,'Accuracy on test ',accu)
        else:
            print('Epoch', epoch + 1, ' completed out of', epochs, 'loss:', epoch_loss)
    session.close()
    return pred,accu


from dataset_Reader import read_cifar10
data = read_cifar10('/home/prem/Desktop/cifer-10/cifar-10-batches-py/', one_hot=True)
train_x=data['train_x']
train_y=data['train_y']
test_x=data['test_x']
test_y=data['test_y']
text_labels=data['text_labels']
#from sklearn import decomposition
#pca = decomposition.PCA(n_components=768)
#pca.fit(train_x)
#utils.save_model(pca,"pca.pkl")
#pca=utils.load_model("pca.pkl")
#train_x = pca.transform(train_x)
#test_x = pca.transform(test_x)
print (train_x.shape, train_y.shape, test_x.shape, test_y.shape)
max = np.max(train_x)
min = np.min(train_x)
mean = np.mean(train_x)
train_x = (train_x - mean) / (max - min)
test_x = (test_x - mean) / (max - min)

pred,accuracy=cnn(train_x, train_y, test_x, test_y)
print ("Accuracy : ",(accuracy))
print ("Test DATA Prediction ",(pred))
cm=utils.get_confusion_matrix(actual=utils.back_from_onehot(test_y[0:500,]),
                               pred=pred,n_classes=10,class_names=text_labels)
print ("Confusion Matrix")
print (cm)
utils.plot_confusion_matrix(cm, classes=text_labels,title='Confusion matrix, without normalization')
