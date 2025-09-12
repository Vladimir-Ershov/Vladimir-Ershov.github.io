import matplotlib.pyplot as plt

from turtle import forward
import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('heart.csv')

print(df.columns)
print(df.shape)

X = np.array(df.loc[:, df.columns != 'output'])
y = np.array(df['output'])

print(f"X {X.shape} , Y {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

class SingleLL:
    def __init__(self, shape):
        self.w = np.random.randn(shape)
        self.b = 0
        def act_sig(x):
            x_arr = np.asarray(x)
            result = np.clip(1 / (1 + np.exp(-x_arr)), 0.0001, 0.9999)
            return result.item() if np.isscalar(x) else result
        
        self.act = act_sig
        def d_act(x):
            x_arr = np.asarray(x)
            act = act_sig(x_arr)
            grad = act * (1 - act)
            grad = np.where((act <= 0.0001) | (act >= 0.9999), 0.0, grad)
            return grad.item() if np.isscalar(x) else grad
        self.d_act = d_act
        self.cache = 0
        self.cache_x = 0
        

    def forward(self, x):
        self.cache_x = x
        self.cache = self.act(np.sum(np.dot(x,self.w)) + self.b)
        return self.cache


class NN:
    def __init__(self, layers, LR):
        self.layers = layers
        self.LR = LR
        loss_f = lambda y, y_pred: (y-y_pred)*(y-y_pred)
        self.loss = loss_f
        self.stats = []
        self.n = len(layers)

    def forward(self, x, layer=0):
        if layer == len(self.layers):
            return x
        else:
            return self.forward(self.layers[layer].forward(x), layer+1)


    # dL/dw = dl/dyp * dyp/dact * dact/dw
    def optimise(self, y_true, y_pred):
        dnext = 2 * (y_true - y_pred)
        for i in range(self.n):
            layer_cur = self.layers[-i-1]
            d_act = layer_cur.d_act(layer_cur.cache_x)
            d_l_d_w = dnext * d_act * layer_cur.cache_x
            dnext = dnext * d_act * layer_cur.w
            layer_cur.w += d_l_d_w * self.LR

    
    def train(self, x_train, x_test, y_train, y_test, epoch=1000):
        self.train_loss = []
        self.test_loss = [] 
        for ep in range(epoch):
            perm = np.random.permutation(len(x_train))
            batch_x = x_train[perm]
            batch_y = y_train[perm]
            batch_loss = 0
            for i in range(len(batch_x)):
                y_pred_i = self.forward(batch_x[i])
                batch_loss += self.loss(batch_y[i], y_pred_i)
                self.optimise(batch_y[i], y_pred_i)
            print(f"{ep}: \t\t train_loss = {batch_loss}")
            self.train_loss.append(batch_loss)
            
            test_loss = 0
            for i in range(len(x_test)):
                y_pred_i = self.forward(x_test[i])
                print(np.round(y_pred_i))
                test_loss += self.loss(y_test[i], y_pred_i)
            print(f"{ep}: \t\t test_loss = {test_loss}")
            self.test_loss.append(test_loss)
        print("Train finisied!")
        self.tn = len(self.test_loss)

nn = NN([SingleLL(X_train.shape[1]) for _ in range(1)], 0.01)

nn.train(X_train_scale, X_test_scale, y_train, y_test, epoch = 10)

sns.lineplot(x=list(range(nn.tn)), y=nn.test_loss, label="Test Loss")
sns.lineplot(x=list(range(nn.tn)), y=nn.train_loss, label="Train Loss")
plt.show()
