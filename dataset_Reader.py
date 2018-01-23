import pickle
import numpy as np
def save_model(model,filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
def load_model(filename):
    pickle_in = open(filename, 'rb')
    return pickle.load(pickle_in)
def load_encoding_model(filename,encode='ascii'):
    pickle_in = open(filename, 'rb')
    return pickle.load(pickle_in,encoding=encode)
def convert_to_onehot(y_,n_classes):
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]
def read_cifar10(path,one_hot=True):
    import os
    text_labels=load_model(path+'batches.meta')['label_names']
    if os.name == 'nt':
        for i in range(5):
            data=load_encoding_model(path + 'data_batch_'+str(i+1),encode='bytes')
            if i==0:
                train_x=data[b'data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3)
                train_y=np.array(data[b'labels']).reshape(10000,1)
                continue
            train_x= np.vstack((train_x,data[b'data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3)))
            train_y = np.vstack((train_y,np.array(data[b'labels']).reshape(10000,1)))
        data=load_encoding_model(path +'test_batch',encode='bytes')
        test_x=data[b'data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3)
        test_y=np.array(data[b'labels']).reshape(10000,1)
        if one_hot == True:
            train_y=convert_to_onehot(train_y,10)
            test_y = convert_to_onehot(test_y, 10)
    else:
        for i in range(5):
            data=load_model(path + 'data_batch_'+str(i+1))
            if i==0:
                train_x=data['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3)
                train_y=np.array(data['labels']).reshape(10000,1)
                continue
            train_x= np.vstack((train_x,data['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3)))
            train_y = np.vstack((train_y,np.array(data['labels']).reshape(10000,1)))
        data=load_model(path +'test_batch')
        test_x=data['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2).reshape(-1, 32*32*3)
        test_y=np.array(data['labels']).reshape(10000,1)
        if one_hot == True:
            train_y=convert_to_onehot(train_y,10)
            test_y = convert_to_onehot(test_y, 10)
    return {'train_x':train_x,'train_y':train_y,'test_x':test_x,'test_y':test_y,'text_labels':text_labels}

#path='/home/prem/Desktop/cifer-10/cifar-10-batches-py/'
#data=read_cifar10(path,False)
