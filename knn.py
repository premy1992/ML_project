from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import utils
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

### Checking DATAset shape
print train_x.shape, train_y.shape, test_x.shape, test_y.shape
train_y=train_y.reshape(50000)
test_y=test_y.reshape(10000)
print train_x.shape, train_y.shape, test_x.shape, test_y.shape
#creating a classifer Object
clf = KNeighborsClassifier(n_neighbors=50,n_jobs=-1)
print "Training ...... "
#clf.fit(train_x,train_y)

# clf.fit(X_transformed,cifar10_labels)
# print "start to predict"
# pred_labels = clf.predict(X_test_transformed)
# print np.mean(pred_labels == cifar10_test_label)

#saving trained model
#utils.save_model(clf,"knn_cifar10.pickle")
clf=utils.load_model("knn_cifar10.pickle")

pred=clf.predict(test_x)
acc=get_accuracy(test_y,pred)


print "Accuracy : ",(acc)
print "Test DATA Prediction ",(pred)
cm=utils.get_confusion_matrix(actual=test_y,
                               pred=pred,n_classes=10,class_names=text_labels)
print "Confusion Matrix"
print cm
utils.plot_confusion_matrix(cm, classes=text_labels,title='Confusion matrix, without normalization')
