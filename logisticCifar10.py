import tensorflow as tf
import numpy as np

def train_model(train_x, train_y, feature_size,n_class, test_x=None , test_y=None ,learning_rate=0.001,epochs=1, datatype=tf.float32, batch_size=None):
    if batch_size == None: batch_size = train_x.shape[0]

    w = tf.Variable(tf.random_normal(dtype=datatype, shape=[feature_size, n_class],mean=0,stddev=1))
    b = tf.Variable(tf.random_normal(dtype=datatype,shape=[1,n_class]))
    x = tf.placeholder(dtype=datatype, shape=[None, feature_size])
    y = tf.placeholder(dtype=datatype, shape=[None, n_class])

    hypothesis = tf.add(tf.matmul(x, w), b)
    activation=tf.nn.softmax(hypothesis, name="activation")
    cost = tf.nn.l2_loss(activation - y, name="squared_error_cost")

    learningRate = tf.train.exponential_decay(learning_rate=learning_rate,
                                              global_step=1,
                                              decay_steps=train_x.shape[0],
                                              decay_rate=0.95,
                                              staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learningRate)
    train = optimizer.minimize(cost)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    pred=None
    accu=None
    for epoch in range(epochs):
        epoch_loss = 0
        i = 0
        while i < len(train_x):
            start = i
            end = i + batch_size
            if (end > len(train_x)): end = len(train_x)
            batch_x = np.array(train_x[start:end])
            try: batch_x=batch_x.reshape(end - start, feature_size)
            except: print("check train_x shape");
            batch_y = np.array(train_y[start:end])
            try: batch_y=batch_y.reshape(end - start, n_class)
            except: print("check train_y shape");
            _, c = session.run([train, cost], feed_dict={x: batch_x, y: batch_y})
            epoch_loss += c
            i += batch_size
        if test_x is not None and test_y is not None:
            test_x=test_x.reshape(len(test_x),feature_size)
            test_y = test_y.reshape(len(test_y),n_class)
            prediction= tf.argmax(activation, 1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y, 1)), "float"))
            pred,accu=session.run([prediction,accuracy],feed_dict={x: test_x, y: test_y})
            print 'Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss,'Accuracy on test',accu
        else:
            print('Epoch', epoch + 1, ' completed out of', epochs, 'loss:', epoch_loss)
    W = session.run(w)
    B = session.run(b)
    session.close()
    return [W, B],pred,accu
def prediction(model,data):
    import numpy as np
    return np.argmax(data.dot(model[0]) + model[1],axis=1)
def get_accuracy(actual,pred):
    return float(np.sum(np.equal(pred,actual)))/float(len(actual))*100


import dataset_reader,utils
data = dataset_reader.read_cifar10('cifar-10-batches-py/', one_hot=True)
train_x=data['train_x']
train_y=data['train_y']
test_x=data['test_x']
test_y=data['test_y']
text_labels=data['text_labels']
model,pred,accuracy =train_model(train_x,train_y,3072,n_class=10,test_x=test_x,test_y=test_y,
                                 epochs=2500,batch_size=100,learning_rate=0.0001)
utils.save_model(model,"LogisticCIfar10.pickle")
model=utils.load_model("LogisticCIfar10.pickle")
pred=prediction(model=model,data=test_x)
accuracy=get_accuracy(actual=utils.back_from_onehot(test_y),pred=pred)
print (accuracy)
print (pred)
cm=utils.get_confusion_matrix(actual=utils.back_from_onehot(test_y),pred=pred,n_classes=3,class_names=text_labels)
utils.plot_confusion_matrix(cm, classes=text_labels,title='Confusion matrix, without normalization')
print cm
