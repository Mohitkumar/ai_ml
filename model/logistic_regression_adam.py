import torch

class AdamRegression:
    def __init__(self, num_classes, num_features, device='cpu') -> None:
        self.device = device
        self.w = torch.zeros(size=[num_features, num_classes],requires_grad=True, device=self.device)
        self.b = torch.zeros(size=[num_classes,1], requires_grad=True, device = self.device)
        self.v_dw = torch.zeros(size=[num_features, num_classes], device=self.device)
        self.v_db = torch.zeros(size=[num_classes, 1], device=self.device)
        self.s_dw = torch.zeros(size=[num_features, num_classes], device=self.device)
        self.s_db = torch.zeros(size=[num_classes, 1], device=self.device)

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
    
    def _forward_regularized(self, w, b, X, Y, lambd):
        f = w.T @ X + b
        A = self._sigmoid(f)
        cost_inner = Y * torch.log(A) + (1 - Y) * torch.log(1-A)
        l2_reg = (lambd / (2 * X.shape[1])) * torch.sum(torch.square(w))
        cost = -1 * (cost_inner.sum() / X.shape[1])  + l2_reg
        cost.backward()
        result = cost,w.grad,b.grad
        return result


    def optimize(self, X, Y, num_iterations=100, learning_rate=0.009, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, lambd=0.0, print_cost=False):
        X = self._move_to_device(X)
        Y = self._move_to_device(Y)
        t = 0
        for i in range(num_iterations):
            t = t + 1
            if(lambd == 0.0):
                cost, dw, db = self._forward(self.w,self.b,X,Y)
            else:
                cost, dw, db = self._forward_regularized(self.w,self.b,X,Y, lambd) 
            self.v_dw = (beta1 * self.v_dw + (1-beta1) * dw)
            self.v_db = (beta1 * self.v_db + (1-beta1) * db) 
            v_dw_corrected = self.v_dw / (1- (beta1 ** t))
            v_db_corrected = self.v_db / (1- (beta1 ** t))
            self.s_dw = (beta2 * self.s_dw ) + (1- beta2) * (dw ** 2)
            self.s_db = ((beta2 * self.s_db ) + (1- beta2) * (db ** 2))
            s_dw_corrected = self.s_dw / (1- (beta2 ** t))
            s_db_corrected = self.s_db  / (1- (beta2 ** t))
            with torch.no_grad():
                self.w -= (learning_rate * (v_dw_corrected / (torch.sqrt(s_dw_corrected)+epsilon)))
                self.b -= (learning_rate * (v_db_corrected / (torch.sqrt(s_db_corrected)+epsilon)))
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
        return ((self.predict(X) == Y_decoded).sum() / len(Y_decoded)) * 100.0