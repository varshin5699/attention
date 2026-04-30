from typing import Optional, Union, Any
import torch
import torch.nn as nn
from softmax import Softmax
from silu import SiLU, L2Norm
class Attention:
    def __init__(self, X: Union[torch.Tensor, list[float]], **kwargs):

        ## configs
        self.type = kwargs.get("attention_type","gqa")
        self.dk = kwargs.get("dk",64)
        self.dv = self.dk
        self.dq = self.dk
        self.dm = kwargs.get("dm", 512)
        # self.dh = kwargs.get("dh", 8)
        # self.g = kwargs.get("num_groups", 2)
        # assert self.dm % self.dh == 0
        # assert self.dh % self.g == 0

        ## Matrices
        self.X = X #[batch_dim, seq_len (N), model_dim]
        # self.Wk #for key computation, [dm, dk] for mha, dm % dh ==0; for gqa, dm%(dh//g)==0
        # self.Wv #for value computation, [dm, dv]
        self.Wq = torch.rand(self.dm, self.dq)#for query computaiton, [dm, dq]
        self.Wo = torch.rand(self.dm, self.dm) #after head computation, [dm, dm]
    
    def forward():
        return
    
class GatedDeltaNetAttention(Attention):
    
    def __init__(self,):
        super().__init__()
        self.Wk = nn.Linear(self.dm, self.dk)
        self.Wv = nn.Linear(self.dm, self.dv)
        self.Wq = nn.Linear(self.dm, self.dq)
        self.Wo = nn.Linear(self.dk, self.dm)
        # self.alpha = nn.Linear(self.dm, self.dm) # for decay of the state matrix, [dm, dm]
        # self.beta = nn.Linear(self.dm, self.dm) # learning rate [dm, dm]

        ## for efficiency, having both alpa and beta in the same matrix
        self.alp_bet = nn.Linear(self.dm, self.dk*2) # [dm, dm*2]
        self.gate = nn.Linear(self.dm, self.dm) 
        self.state_matrix = torch.zeros(self.dk, self.dk) # [dk , dk]
        # TODO: add depthwise convolution to ensure more inductive bias since the state matrix doesn't store information
        # about the position of the token.
        # self.short_conv = nn.Conv1d
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # steps in Gated DeltaNet Attention:
        # 1. compute k,q,v values
        key = self.Wk(X) # [batch_dim, seq_len, dk]
        value = self.Wv(X) # [batch_dim, seq_len, dv]
        query = self.Wq(X) # [batch_dim, seq_len, dq]

        #2. do short convolution / depthwise convolution on X

        #3. Apply SILU on X (short conv)
        ## This is done to emulate the sparsity that softmax does - except its chosen to ensure phi(K).phi(Q) = softmax(KQ^T)
        ## Usually 1+ELU is chosen since its strictly positive like softmax (since negative values were believed to lose information), while silu does take a slight bound with negative values,
        ## it works well for linear attention, since there is more interference of signals in the lossy state matrix in linear attention
        key = SiLU(key)
        query = SiLU(query)
        value = SiLU(value)


        #4. Apply L2 Norm on the QK vectors to ensure the stability of state matrix -> the reason for this is that the eigen value of 
        # (I-b*KK^T) are 1-b*||K||^2 along the direction of K, and 1 along the rest of eigen vectors, thus if ||K||^2 is large,
        # the state matrix would blow up the values along the direction of K, and this can lead to the error in retrieving the value 
        # corresponding to K higher, which is undesirable. Ideally, we want the eigen value to be 0, which would requre ||K||^2 to be 1/b (1).
        # Reason for Not L1: L1 norm only ensures the components add to 1, but not the length of the K vector. Hence, L2 is more apt.
        query = L2Norm(query)
        key = L2Norm(key) ## actually we cant use this L2Norm during training, because the gradient of this isn't computed by the
        ## autograd, and not including can lead to blowing up of loss, and nullify the effects of gradient updates.
        
        # 5. compute the state matrix as the outer product of K and V
        state_update = torch.einsum("bik,bij->bkj", key, value) # [batch_dim, dk, dv]

        #6. compute alpa, beta for the X and update the state matrix  
        alpha,beta = self.alp_bet(X).chunk(2, dim=-1) # [batch_dim, L, dk], [batch_dim, L, dk]

        gate = SiLU(self.gate(X)) # [batch_dim, L, dm]

        #7. update the state matrix with the delta rule applied
        self.state_matrix = self.state_matrix @ (alpha - beta @ (key.transpose(-2,-1)@key)) + state_update # [dk, dk] @ [batch_dim, dm, dm] -> [batch_dim, dm, dk] -> [batch_dim, dk, dk]

        #8. compute the output as product of state and query
        retrieval = query @ self.state_matrix # [batch_dim, seq_len, dk] @ [batch_dim, dk, dk] -> [batch_dim, seq_len, dk]
       
       #9. apply the RMSNorm, and then the gate to the retrieval
        rmsnorm = nn.RMSNorm(self.dk)
        retrieval = rmsnorm(retrieval) * gate # [batch_dim, seq_len, dk]
    
        #10. linear transformation to get the output
        attn_output = self.Wo(retrieval) # [batch_dim, seq_len, dm]
        return attn_output

## TODO: instead of inheriting, another way would be to simply override the value of the number of groups to be 1 for mha, and dh for mqa
## and just maintain one gqa implementation, thereby reducing density of code.
## Here, I went ahead and implemented gqa and mha seeking clarity
class MultiHeadAttn(Attention):
    def __init__(self,**kwargs):
        super().__init__()
        self.h = kwargs.get("num_heads", 8)
        assert self.dk % self.h == 0
        self.dh = self.dk//self.h
        self.Wk = torch.rand(self.dm, self.dk)
        self.Wv = torch.rand(self.dm, self.dv)
        
    def forward(self):
        ## steps in MHA:
        # 1. compute k,q,v values
        key = self.X@(self.Wk.transpose(-2,-1)) #from [batch_dim, seq_len, dk] -> [batch_dim, dh, seq_len, dk//dh]
        value = self.X@(self.Wv.transpose(-2, -1))
        query = self.X@(self.Wq.transpose(-2,-1))

        # 2. reshaping to get the head
        # for reshaping we could use both view and reshape but view needs the tensor object to be contiguous in memory, while reshape can handle non-contiguous tensors.
        key = key.reshape(key.shape[0], key.shape[1], self.h, self.dh).transpose(1,2) # [batch_dim, h, seq_len, dh]
        value = value.reshape(value.shape[0], value.shape[1], self.h, self.dh).transpose(1,2) # [batch_dim, h, seq_len, dh]
        query = query.reshape(query.shape[0], query.shape[1], self.h, self.dh).transpose(1,2) # [batch_dim, h, seq_len, dh

        # 3. performing kq^T

        attn_weights = query @ key.transpose(-2,-1) / (self.dh**0.5) # [batch_dim, h, seq_len, seq_len]

        # 4. apply causal mask
        mask = torch.tril(torch.ones(attn_weights.shape[-2:]), diagonal=1).bool() # [seq_len, seq_len]
        attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        # 5. apply softmax
        softmax = Softmax(safe=True)
        attn_weights = softmax.safe_softmax(attn_weights, dim=-1) # [batch_dim, h, seq_len, seq_len]

        # 6. matmul with V_i, i being the head number
        attn_output = attn_weights @ value # [batch_dim, h, seq_len, dh]

        # 7. reshape and linear transformation
        attn_output = attn_output.transpose(1,2).reshape(attn_output.shape[0], attn_output.shape[2], self.dk) # [batch_dim, seq_len, dk]
        attn_output = attn_output @ self.Wo # [batch_dim, seq_len, dm]

        return attn_output, attn_weights
    
class GQAttention(Attention):
    ## MQA is the case where g=h, and MHA is the case where g=1, thus we can maintain one implementation for GQA and just change the value of g to get the other two types of attention.
    def __init__(self,**kwargs):
        super().__init__()
        self.g = kwargs.get("num_groups", 2)
        self.kv_heads = self.h//self.g
        assert self.dm % self.g == 0
        assert self.dh % self.g == 0
        self.Wk = torch.rand(self.dm, self.dk)
        self.Wv = torch.rand(self.dm, self.dv)

    def repeat_kv(self, X: torch.Tensor, g: int) -> torch.Tensor:

        ## This function is also the overhead caused by GQA.
        ## It tries to recompute the key, value for the remaining heads thereby trading-off compute for memory.

        X = X.reshape(X.shape[0], X.shape[1], self.kv_heads, self.dh*self.g) # [batch_dim, seq_len, kv_heads, dh*g]
        bs, slen, n_kv_heads, head_dim = X.shape
        if g ==1: 
            return X
        else:
            X = X.unsqueeze(2).expand(bs, slen, self.g, n_kv_heads, head_dim) # [batch_dim, seq_len, g, kv_heads, dh*g]
            X = X.reshape(bs, slen, self.h, self.dh) # [batch_dim, seq_len, h, dh]
            return X.transpose(1,2) # [batch_dim, h, seq_len, dh]

    def forward(self):
    ## steps in GQA:
        # 1. compute k,q,v values
        key = self.X@(self.Wk.transpose(-2,-1)) #from [batch_dim, seq_len, dk] -> [batch_dim, dh, seq_len, dk//dh]
        value = self.X@(self.Wv.transpose(-2, -1))
        query = self.X@(self.Wq.transpose(-2,-1))

        # 2. reshaping to get the head
        # for reshaping we could use both view and reshape but view needs the tensor object to be contiguous in memory, while reshape can handle non-contiguous tensors.
        key = self.repeat_kv(key, self.g) # [batch_dim, h, seq_len, dh]
        value = self.repeat_kv(value, self.g) # [batch_dim, h, seq_len, dh]
        query = query.reshape(query.shape[0], query.shape[1], self.h, self.dh).transpose(1,2) # [batch_dim, h, seq_len, dh]

        # The major change required in GQA would be during the reshaping; 
        # the dh for kv would be dh //g, while for q it would be dh, and the number of heads would be h//g.
        # thus, we would have to expand the key and value tensors to first add another dimension for the group, 
        # and then reshape to get to multiply and reduce it to 4 dimensions as in MHA.

        # 3. performing kq^T

        attn_weights = query @ key.transpose(-2,-1) / (self.dh**0.5) # [batch_dim, h, seq_len, seq_len]

        # 4. apply causal mask
        mask = torch.tril(torch.ones(attn_weights.shape[-2:]), diagonal=1).bool() # [seq_len, seq_len]
        attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        # 5. apply softmax
        softmax = Softmax(safe=True)
        attn_weights = softmax.safe_softmax(attn_weights, dim=-1) # [batch_dim, h, seq_len, seq_len]

        # 6. matmul with V_i, i being the head number
        attn_output = attn_weights @ value # [batch_dim, h, seq_len, dh]

        # 7. reshape and linear transformation
        attn_output = attn_output.transpose(1,2).reshape(attn_output.shape[0], attn_output.shape[2], self.dk) # [batch_dim, seq_len, dk]
        attn_output = attn_output @ self.Wo # [batch_dim, seq_len, dm]

        return attn_output, attn_weights
    



