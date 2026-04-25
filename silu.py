import torch
def SiLU(X: torch.Tensor)->torch.Tensor:
    return X/(1+(-X).exp())

def L2Norm(X: torch.Tensor)->torch.Tensor:
    return X/(X.norm(dim=-1, keepdim=True)+1e-8)