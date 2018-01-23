import numpy as np
import tensorflow as tf
class pca:
    def __init__(self):
        self.u_reduce=None
        return
    def train(self,data,component=None,variance_retained=0.995):
        data=np.array(data)
        covariance_mat=(data.transpose()).dot(data)/data.shape[1]
        u, s, v = np.linalg.svd(covariance_mat, full_matrices=True,)
        n = s.shape[0]
        if(component!=None):
            self.u_reduce = u[:, 0:component]
            if component == n: self.u_reduce=np.ones((n, n))
        else:
            deno=0
            for i in range(n):
                deno=deno+s[i]
            num=0
            for k in range(n):
                num=num+s[k]
                if((num/deno)>variance_retained):
                    component=k+1
                    break;
            if component==n: self.u_reduce=np.ones((n,n))
            self.u_reduce = u[:, 0:component]
    def transform(self,data):
        return data.dot(self.u_reduce)
'''
from dataset_reader import read_iris
train_x, train_y, test_x, test_y = read_iris(one_hot=False)
p=pca()
u=p.train(train_x,component=None,variance_retained=0.995)
train_x=p.transform(train_x)
test_x=p.transform(test_x)
print train_x.shape,train_y.shape
'''