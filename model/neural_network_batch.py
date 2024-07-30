import torch

class Model:
    def __init__(self, layer_dims, device='cpu', adam=False) -> None:
        self.device = device
        self.layer_dims = layer_dims
        self.adam = adam
        self.parameters = {}
        L = len(layer_dims)
        for l in range(1,L):
            W = torch.rand(size=[layer_dims[l],layer_dims[l-1]], device=self.device) * 0.001
            b = torch.rand(size=[layer_dims[l],1], device=self.device) * 0.001
            self.parameters['w' + str(l)] = W.clone().detach().requires_grad_(True)
            self.parameters['b' + str(l)] = b.clone().detach().requires_grad_(True)
            if adam:
                self.parameters['v_dw' + str(l)] = torch.zeros(size=[layer_dims[l],layer_dims[l-1]], device=self.device)
                self.parameters['v_db' + str(l)] = torch.zeros(size=[layer_dims[l], 1], device=self.device)
                self.parameters['s_dw'+ str(l)] = torch.zeros(size=[layer_dims[l],layer_dims[l-1]], device=self.device)
                self.parameters['s_db' + str(l)] = torch.zeros(size=[layer_dims[l], 1], device=self.device)

    def _move_to_device(self,tensor):
            if self.device != 'cpu':
                return tensor.to(self.device)
            else:
                return tensor
        
    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    

    def _linear_forward(self, A_prev, w, b, activation):
        if activation == 'relu':
            Z = w @ A_prev + b
            A = torch.nn.functional.relu(Z)
        else:
            Z = w @ A_prev + b
            A = self._sigmoid(Z)    
        return A

    def _forward(self, X, Y):
        A = X
        L = len(self.layer_dims) - 1
        for l in range(1, L):
            A_prev = A
            A = self._linear_forward(A_prev, self.parameters['w' + str(l)], self.parameters['b' + str(l)],"relu")
        AL = self._linear_forward(A, self.parameters['w' + str(L)], self.parameters['b' + str(L)],"sigmoid")
        cost_inner = Y * torch.log(AL) + (1 - Y) * torch.log(1-AL)
        cost = -1 * (cost_inner.sum() / Y.shape[1])
        cost.backward()
        return cost
    
    
    def optimize(self, X, Y, batch_size = 500, num_iterations=100, learning_rate=0.009, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,print_cost=False):
        X = self._move_to_device(X)
        Y = self._move_to_device(Y)
        t = 0
        total_cost = 0.0
        for i in range(num_iterations):
            t = t + 1
            for batch_i in range(0, Y.shape[1],batch_size):
                X_batch = X[:, batch_i:batch_i+batch_size]
                Y_batch = Y[:, batch_i:batch_i+batch_size]
                cost = self._optimize_batch(X_batch, Y_batch, learning_rate, beta1, beta2,epsilon)
                total_cost += cost

            total_cost = total_cost / batch_size
            if i % 100 == 0:
                if print_cost:
                    accuracy = self.accuracy(X,Y)
                    print("iteration={}, cost={:.2f}, accuracy={:.2f}%".format(i, total_cost, accuracy))        

    def _optimize_batch(self, X, Y, learning_rate=0.009, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        cost = self._forward(X,Y)
        L = len(self.layer_dims)
        for l in range(1,L):
            if self.adam:
                self.parameters['v_dw'+str(l)] = (beta1 * self.parameters['v_dw'+str(l)]+ (1-beta1) * self.parameters['w'+str(l)].grad)
                self.parameters['v_db'+str(l)] = (beta1 * self.parameters['v_db'+str(l)]+ (1-beta1) * self.parameters['b'+str(l)].grad)
                self.parameters['v_dw_corrected'+str(l)] = self.parameters['v_dw'+str(l)] / (1- (beta1 ** t))
                self.parameters['v_db_corrected'+str(l)] = self.parameters['v_db'+str(l)] / (1- (beta1 ** t))
                self.parameters['s_dw'+str(l)] = (beta2 * self.parameters['s_dw'+str(l)]) + (1- beta2) * (self.parameters['w'+str(l)].grad ** 2)
                self.parameters['s_db'+str(l)] = (beta2 * self.parameters['s_db'+str(l)]) + (1- beta2) * (self.parameters['b'+str(l)].grad ** 2)
                self.parameters['s_dw_corrected'+str(l)] = self.parameters['s_dw'+str(l)] / (1- (beta2 ** t))
                self.parameters['s_db_corrected'+str(l)] = self.parameters['s_db'+str(l)] / (1- (beta2 ** t))
                with torch.no_grad():
                    self.parameters['w'+str(l)] -= (learning_rate * (self.parameters['v_dw_corrected'+str(l)]/ (torch.sqrt(self.parameters['s_dw_corrected'+str(l)])+epsilon)))
                    self.parameters['b'+str(l)] -= (learning_rate * (self.parameters['v_db_corrected'+str(l)] / (torch.sqrt(self.parameters['s_db_corrected'+str(l)])+epsilon)))
                    self.parameters['w'+str(l)].grad = None
                    self.parameters['b'+str(l)].grad = None   
            else:
                with torch.no_grad():
                    self.parameters['w'+str(l)] -= (learning_rate * self.parameters['w'+str(l)].grad)
                    self.parameters['b'+str(l)] -= (learning_rate * self.parameters['b'+str(l)].grad)
                    self.parameters['w'+str(l)].grad = None
                    self.parameters['b'+str(l)].grad = None     
        return cost
    
    def predict(self, X):
        X = self._move_to_device(X)
        A = X
        L = len(self.layer_dims) -1
        for l in range(1, L):
            A_prev = A
            A = self._linear_forward(A_prev, self.parameters['w' + str(l)], self.parameters['b' + str(l)],"relu")
        AL = self._linear_forward(A, self.parameters['w' + str(L)], self.parameters['b' + str(L)],"sigmoid")
        return torch.argmax(AL,dim=0)
    
    def accuracy(self, X, Y):
        X = self._move_to_device(X)
        Y = self._move_to_device(Y)
        Y_decoded = torch.argmax(Y, dim=0)
        return ((self.predict(X) == Y_decoded).sum() / len(Y_decoded)) * 100.0