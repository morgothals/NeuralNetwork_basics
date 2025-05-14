# complex_mlp.optimizer.py
import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        # minden Dense rétegen frissítjük W, b értékeit
        for layer in layers:
            if hasattr(layer, "dW"):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.vW = []
        self.vb = []
        self.initialized = False

    def step(self, layers):
        # egyszer inicializáljuk a tartalékokat
        if not self.initialized:
            for layer in layers:
                if hasattr(layer, "dW"):
                    self.vW.append(np.zeros_like(layer.W))
                    self.vb.append(np.zeros_like(layer.b))
            self.initialized = True

        idx = 0
        for layer in layers:
            if hasattr(layer, "dW"):
                # momentum update
                self.vW[idx] = self.momentum*self.vW[idx] + (1-self.momentum)*layer.dW
                self.vb[idx] = self.momentum*self.vb[idx] + (1-self.momentum)*layer.db
                layer.W -= self.lr * self.vW[idx]
                layer.b -= self.lr * self.vb[idx]
                idx += 1

class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.sW = []
        self.sb = []
        self.initialized = False

    def step(self, layers):
        if not self.initialized:
            for layer in layers:
                if hasattr(layer, "dW"):
                    self.sW.append(np.zeros_like(layer.W))
                    self.sb.append(np.zeros_like(layer.b))
            self.initialized = True

        idx = 0
        for layer in layers:
            if hasattr(layer, "dW"):
                # squared gradient average
                self.sW[idx] = self.beta*self.sW[idx] + (1-self.beta)*(layer.dW**2)
                self.sb[idx] = self.beta*self.sb[idx] + (1-self.beta)*(layer.db**2)
                layer.W -= self.lr * layer.dW / (np.sqrt(self.sW[idx]) + self.eps)
                layer.b -= self.lr * layer.db / (np.sqrt(self.sb[idx]) + self.eps)
                idx += 1

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mW = []
        self.vW = []
        self.mb = []
        self.vb = []
        self.t = 0
        self.initialized = False

    def step(self, layers):
        if not self.initialized:
            for layer in layers:
                if hasattr(layer, "dW"):
                    self.mW.append(np.zeros_like(layer.W))
                    self.vW.append(np.zeros_like(layer.W))
                    self.mb.append(np.zeros_like(layer.b))
                    self.vb.append(np.zeros_like(layer.b))
            self.initialized = True

        self.t += 1
        idx = 0
        for layer in layers:
            if hasattr(layer, "dW"):
                gW = layer.dW
                gb = layer.db
                # 1st moment
                self.mW[idx] = self.beta1*self.mW[idx] + (1-self.beta1)*gW
                self.mb[idx] = self.beta1*self.mb[idx] + (1-self.beta1)*gb
                # 2nd moment
                self.vW[idx] = self.beta2*self.vW[idx] + (1-self.beta2)*(gW**2)
                self.vb[idx] = self.beta2*self.vb[idx] + (1-self.beta2)*(gb**2)
                # bias correction
                mW_hat = self.mW[idx] / (1 - self.beta1**self.t)
                mb_hat = self.mb[idx] / (1 - self.beta1**self.t)
                vW_hat = self.vW[idx] / (1 - self.beta2**self.t)
                vb_hat = self.vb[idx] / (1 - self.beta2**self.t)
                # update
                layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
                layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)
                idx += 1
