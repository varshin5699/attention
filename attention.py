from typing import Optional, Union, Any
import torch
from softmax import Softmax
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
    



