import torch
class Softmax:
    def __init__(self, safe: bool):
        self.input = None
        self.output = None
        self.safe = safe

    def safe_softmax(self, X: torch.Tensor,dim: int)->torch.Tensor:
        if self.safe:
            ## scale it by subtracting the maximum value of X
            X = X-torch.max(X)
        ## calculate the softmax across the given dimension
        self.output = X.exp().divide(X.exp().sum(dim = dim, keepdim=True))
        return self.output

    
    
    
        