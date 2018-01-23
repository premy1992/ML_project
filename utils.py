import numpy as np
def reduce_resolution(source, destination, width=80, length=80):
    from PIL import Image
    from resizeimage import resizeimage
    with open(source, 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, [width, length])
            cover.save(destination, image.format)
def get_pixel(filename):
    from PIL import Image
    img = Image.open(filename)
    arr = np.array(img)
    return arr
def merge_rgb(pixel):
    return pixel[0] + 256 * pixel[1] + 256 * 256 * pixel[2]
def save_model(model,filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
def load_model(filename):
    import pickle
    pickle_in = open(filename, 'rb')
    return pickle.load(pickle_in)
def convert_to_onehot(y,n_class):
    b = np.zeros((len(y),n_class ))
    for i in range(0,len(y)):
        b[i,y[i]] = 1
    return b;
def back_from_onehot(y):
    x=np.argmax(y, axis=1)
    return x
def sigmoid(x):
    import numpy as np
    return 1 / (1 + np.exp(-x))
def get_confusion_matrix(actual,pred,n_classes,class_names,asText=False):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=actual,  # True class for test-set.
                          y_pred=pred)  # Predicted class.
    if asText==True:
        # Print the confusion matrix as text.
        for i in range(n_classes):
            # Append the class-name to each line.
            class_name = "({}) {}".format(i, class_names[i])
            print(cm[i, :], class_name)
    return cm
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix'):
    import matplotlib.pyplot as plt
    import itertools
    import numpy as np
    #cmap=plt.cm.get_cmap('RdBu')
    cmap = plt.cm.get_cmap('Blues')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def feacture_scaling(data,column):
    if column is not None:
        print (2)
