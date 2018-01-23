import tensorflow as tf
import numpy as np
import utils
def train_model(train_x, train_y, feature_size,n_class,hidden_layers,activation_choice, test_x=None , test_y=None
                ,learning_rate=0.001,epochs=1, datatype=tf.float32, batch_size=None):
    if batch_size == None: batch_size = train_x.shape[0]
    dropout_rate=0.7
    def choose_activation_function(node, choice):
        node=tf.nn.dropout(node, dropout_rate)
        if choice == 0:
            return node
        if choice == 1:
            return tf.nn.relu(node)
        if choice == 2:
            return tf.nn.sigmoid(node)
        if choice == 3:
            return tf.nn.tanh(node)
        if choice == 4:
            return tf.nn.dropout(node,dropout_rate)
        if choice == 5:
            return tf.nn.crelu(node)
        if choice == 6:
            return tf.nn.relu6(node)
        if choice == 7:
            return tf.nn.elu(node)
        if choice == 8:
            return tf.nn.softplus(node)
        if choice == 9:
            return tf.nn.softsign(node)

    x = tf.placeholder(dtype=datatype, shape=[None, feature_size])
    y = tf.placeholder(dtype=datatype, shape=[None, n_class])

    hidden_layer_weights_biases={}
    for i in range(len(hidden_layers)):
        if i==0:
            hidden_layer_weights_biases.update({'weights'+str(i):tf.Variable(tf.random_normal([feature_size,hidden_layers[i]])),
                      'biases'+str(i):tf.Variable(tf.random_normal([1,hidden_layers[i]]))})
        else:
            hidden_layer_weights_biases.update({'weights'+str(i):tf.Variable(tf.random_normal([hidden_layers[i-1],hidden_layers[i]])),
                      'biases'+str(i):tf.Variable(tf.random_normal([1,hidden_layers[i]]))})
    output_layer = {'weights':tf.Variable(tf.random_normal([hidden_layers[len(hidden_layers)-1], n_class])),
                    'biases':tf.Variable(tf.random_normal([1,n_class])),}

    layer_output={}
    for i in range(len(hidden_layers)):
        if i==0:
            layer_output.update({str(i):choose_activation_function(tf.add(tf.matmul(x, hidden_layer_weights_biases['weights'+str(i)]),
                                hidden_layer_weights_biases['biases'+str(i)]),activation_choice[i])})
        else:
            layer_output.update({str(i):choose_activation_function(tf.add(tf.matmul(layer_output[str(i-1)], hidden_layer_weights_biases['weights'+str(i)]),
                                hidden_layer_weights_biases['biases'+str(i)]),activation_choice[i])})

    hypothesis = tf.matmul(layer_output[str(len(hidden_layers)-1)], output_layer['weights']) + output_layer['biases']
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y) )

    learningRate = tf.train.exponential_decay(learning_rate=learning_rate,global_step=1,
                                              decay_steps=train_x.shape[0],decay_rate=0.95,staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
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
            try: batch_x=batch_x.reshape(end - start, feature_size)
            except: print("check train_x shape");
            batch_y = np.array(train_y[start:end])
            try: batch_y=batch_y.reshape(end - start, n_class)
            except: print("check train_y shape");
            _, c = session.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            epoch_loss += c
            i += batch_size
    	if test_x is not None and test_y is not None:
            test_x=test_x.reshape(len(test_x),feature_size)
            test_y = test_y.reshape(len(test_y),n_class)
            prediction= tf.argmax(hypothesis, 1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y, 1)), "float"))
            pred,accu=session.run([prediction,accuracy],feed_dict={x: test_x, y: test_y})
            print 'Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss,'Accuracy on test',accu
        else:
            print('Epoch', epoch + 1, ' completed out of', epochs, 'loss:', epoch_loss)
    parameter = session.run(hidden_layer_weights_biases)
    session.close()
    return parameter,pred,accu
def rotate_random_cifar10_images(train_x):
    from PIL import Image
    import random as rn
    for i in range(len(train_x)):
        x = train_x.reshape(60000, 3, 32, 32).transpose(0, 2, 3, 1)
        rotated =Image.Image.rotate(Image.fromarray(x[i]),rn.randint(0, 360))
        x[i]=np.array(rotated)
    return x.reshape(-1,3072)

from dataset_reader import read_cifar10
data = read_cifar10('cifar-10-batches-py/', one_hot=True)
train_x=data['train_x']
train_y=data['train_y']
test_x=data['test_x']
test_y=data['test_y']
text_labels=data['text_labels']
max=np.max(train_x)
min=np.min(train_x)
mean=np.mean(train_x)
std=np.std(train_x)
train_x=(train_x-mean)/(std)
test_x=(test_x-mean)/(std)
print train_x.shape,train_y.shape,test_x.shape,test_y.shape
model,pred,accuracy=train_model(train_x,train_y,3072,n_class=10,
                                hidden_layers=[100,100,100],activation_choice=[2,2,2]
                                ,test_x=test_x, test_y=test_y,epochs=1000,batch_size=5000,learning_rate=0.003)
print "Accuracy : ",(accuracy)
print "Test DATA Prediction ",(pred)
cm=utils.get_confusion_matrix(actual=utils.back_from_onehot(test_y),
                               pred=pred,n_classes=10,class_names=text_labels)
print "Confusion Matrix"
print cm
utils.plot_confusion_matrix(cm, classes=text_labels,title='Confusion matrix, without normalization')

