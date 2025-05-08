import numpy as np

class CrossEntropyLoss:
    def forward(self, preds, targets):
        # preds: (N, C), targets: (N,)
        N = preds.shape[0]
        correct = preds[np.arange(N), targets]
        self.loss = -np.log(correct + 1e-9).mean()
        # store for backward
        self.preds = preds
        self.targets = targets
        return self.loss
    def backward(self, lr):
        grad = self.preds.copy()
        N = grad.shape[0]
        grad[np.arange(N), self.targets] -= 1
        return grad / N

class MSELoss:
    def forward(self, preds, targets):
        # preds: (N, C) or (N,1), targets one-hot or continuous
        self.preds = preds
        self.targets = targets
        self.loss = ((preds - targets)**2).mean()
        return self.loss
    def backward(self, lr):
        N = self.preds.shape[0]
        return 2*(self.preds - self.targets)/N
