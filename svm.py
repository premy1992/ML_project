from sklearn.svm import SVC
import numpy as np
import utils
import pca
def get_accuracy(actual,pred):
    return float(np.sum(np.equal(pred,actual)))/float(len(actual))*100


## loading Cifar10 dataset
from dataset_reader import read_cifar10
data = read_cifar10('cifar-10-batches-py/', one_hot=False)
train_x=data['train_x']
train_y=data['train_y']
test_x=data['test_x']
test_y=data['test_y']
text_labels=data['text_labels']

#p=pca.pca()
#p.train(train_x)
#train_x=p.transform(train_x)
#test_x=p.transform(test_x)
### Checking DATAset shape
print train_x.shape, train_y.shape, test_x.shape, test_y.shape
train_y=train_y.reshape(60000)
test_y=test_y.reshape(10000)

#creating a classifer Object
clf = SVC(C=1.0,decision_function_shape='ovr')
print "Training ...... "
#clf.fit(train_x,train_y)

#saving trained model
#utils.save_model(clf,"svmpca_cifar10.pickle")
clf=utils.load_model("svm_cifar10.pickle")

pred=clf.predict(test_x)
acc=get_accuracy(test_y,pred)


print "Accuracy : ",(acc)
print "Test DATA Prediction ",(pred)
cm=utils.get_confusion_matrix(actual=test_y,
                               pred=pred,n_classes=10,class_names=text_labels)
print "Confusion Matrix"
print cm
utils.plot_confusion_matrix(cm, classes=text_labels,title='Confusion matrix, without normalization')
