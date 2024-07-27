import torch

class LinerRegression:
    def __init__(self, num_classes, num_features, device='cpu') -> None:
        self.device = device
        self.w = torch.zeros(size=[num_features, num_classes],requires_grad=True, device=self.device)
        self.b = torch.zeros(size=[num_classes,1], requires_grad=True, device = self.device)

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _move_to_device(self,tensor):
        if self.device != 'cpu':
            return tensor.to(self.device)
        else:
            return tensor

    def _forward(self, w, b, X, Y):
        f = w.T @ X + b
        A = self._sigmoid(f)
        cost_inner = Y * torch.log(A) + (1 - Y) * torch.log(1-A)
        cost = -1 * (cost_inner.sum(axis=1).sum() / X.shape[1])
        cost.backward()
        result = cost,w.grad,b.grad
        return result

    def optimize(self, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
        X = self._move_to_device(X)
        Y = self._move_to_device(Y)
        for i in range(num_iterations):
            cost, dw, db = self._forward(self.w,self.b,X,Y)
            with torch.no_grad():
                self.w -= (learning_rate * dw)
                self.b -= (learning_rate * db)
                self.w.grad = None
                self.b.grad = None    
            if i % 100 == 0:
                if print_cost:
                    accuracy = self.accuracy(X,Y)
                    print("iteration={}, cost={}, accuracy={}%".format(i, cost, accuracy))

    def predict(self, X):
        X = self._move_to_device(X)
        A = self._sigmoid(self.w.T @ X + self.b)
        return torch.argmax(A,dim=0)
    
    def accuracy(self, X, Y):
        X = self._move_to_device(X)
        Y = self._move_to_device(Y)
        Y_decoded = torch.argmax(Y, dim=0)
        return (self.predict(X) == Y_decoded).sum() / len(Y_decoded)