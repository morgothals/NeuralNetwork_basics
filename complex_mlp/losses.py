# complex_mlp.losses.py
import numpy as np

class CrossEntropyLoss:
    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """
        preds: (N, C) valószínűségek
        targets: (N,) egész címkék
        """
        N = preds.shape[0]
        # numerikusan stabilizáljuk
        clipped = np.clip(preds, 1e-9, 1.0)
        correct = clipped[np.arange(N), targets]
        self.loss = -np.log(correct).mean()

        # backwardhoz elmentjük
        self.preds   = preds
        self.targets = targets
        return self.loss

    def backward(self) -> np.ndarray:
        """
        visszaadja dL/dlogits formájában
        """
        grad = self.preds.copy()
        N = grad.shape[0]
        grad[np.arange(N), self.targets] -= 1
        return grad / N

class MSELoss:
    def forward(self, preds: np.ndarray, targets: np.ndarray) -> float:
        """
        preds: (N, C) vagy (N,1)
        targets: (N, C) vagy (N,1) (one-hot kategória vagy folytonos)
        """
        self.preds   = preds
        self.targets = targets
        self.loss    = np.mean((preds - targets)**2)
        return self.loss

    def backward(self) -> np.ndarray:
        """
        visszaadja dL/dpreds
        """
        N = self.preds.shape[0]
        return 2*(self.preds - self.targets) / N
