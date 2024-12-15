from torch import nn
import torch
from torchsummary import summary

def mask_3d_softmax(score):
    return nn.functional.softmax(score,dim=-1)

class PositionWiseFFN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu1 = nn.GELU()
        self.linear2 = nn.Linear(hidden_size,output_size)
        self.gelu2 = nn.GELU()

    def forward(self,X):
        X = self.gelu1(self.dropout(self.linear1(X)))
        return self.gelu2(self.linear2(X))



class PatchEmbedding(nn.Module):

    def __init__(self,embeding_size:int=128,patch_size:int=32,img_size:int=512,num_channels:int=3):
        super().__init__()
        self.patcher = nn.Conv2d(num_channels,embeding_size,\
                                kernel_size=patch_size,stride=patch_size,)
        self.cls_token = nn.Parameter(torch.randn((1,1, embeding_size))) # 可学习，用于汇聚全局参数
        self.pos_embedding = nn.Parameter(torch.rand((1,(img_size//patch_size)**2 + 1,embeding_size))) # 第0维广播 
        
    def forward(self,X):
        # batch_size, num_channels, height, width
        X = self.patcher(X)
        # batch_size, num_patch, embedding_size
        X = X.reshape(X.shape[0],-1,X.shape[1])
        cls_token = self.cls_token.repeat(X.shape[0],1,1)
        X = torch.cat([cls_token,X],dim=1)
        self.pos_embedding.shape
        return X + self.pos_embedding
    
class Multihead_Attention(nn.Module):
    def __init__(self,num_heads,dropout,query_size,key_size,
                 value_size,hidden_size,bias=False,knowledge_dim=0,**kwargs):
        super().__init__()
        self.knowledge=None
        if knowledge_dim:
            self.knowledge = nn.Parameter(torch.randn(1,knowledge_dim,hidden_size))
        self.num_heads = num_heads
        self.w_q = nn.Linear(query_size,hidden_size,bias=bias)
        self.w_k = nn.Linear(key_size,hidden_size,bias=bias)
        self.w_v = nn.Linear(value_size,hidden_size,bias=bias)
        self.w_o = nn.Linear(hidden_size,hidden_size,bias=bias)
        self.attention = DotproductAttention(dropout)

    def forward(self,query,key,value):

        def transpose_qkv(qkv):
            # qkv.shape:(batch_size,query_size,num_hidden)
            shape = qkv.shape
            # qkv.shape:(batch_size,query_size,num_heads,num_hidden//num_heads)
            qkv = qkv.reshape(shape[0],shape[1],-1,shape[-1]//self.num_heads)
            qkv = qkv.permute(0,2,1,3)
            # 最终qkv.shape:(batch_size*num_heads,query_size,num_hidden//num_heads)
            return qkv.reshape(-1,qkv.shape[-2],qkv.shape[-1])

        if self.knowledge != None:
            key = torch.cat([self.knowledge.repeat(key.shape[0],1,1),key],dim=1)
            value = torch.cat([self.knowledge.repeat(value.shape[0],1,1),value],dim=1)
        
        query = transpose_qkv(self.w_q(query))
        key = transpose_qkv(self.w_k(key))
        value = transpose_qkv(self.w_v(value))

        def transpose_output(output):
            # output.shape:(batch_size*num_heads,query_size,num_hidden//num_heads)
            output = output.reshape(-1,self.num_heads,output.shape[-2],output.shape[-1])
            output = output.permute(0,2,1,3)
            return output.reshape(output.shape[0],output.shape[1],-1)
            
        return self.w_o(transpose_output(self.attention(query,key,value)))
    
class DotproductAttention(nn.Module):
    def __init__(self,dropout,**kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self,query,key,value):
        # score.shape:(batch_size*num_heads,num_query,num_query)
        score = torch.bmm(query,key.transpose(-2,-1))
        self.attention_weight = mask_3d_softmax(score)
        return torch.bmm(self.attention_weight,value)
    
class AddNorm(nn.Module):
    def __init__(self,normshape,dropout,**kwargs):
        super().__init__(**kwargs)
        self.layernorm = nn.LayerNorm(normshape)
        self.dropout = nn.Dropout(dropout)

    def forward(self,X,Y):
        return self.layernorm(X+self.dropout(Y))
    
class Encoder_Block(nn.Module):
    def __init__(self,num_heads,dropout,query_size,key_size,
                 value_size,hidden_size,normshape,ffn_hidden_size,
                 ffn_output_size,**kwargs):
        # query_size=key_size=value_size=hidden_size=normshape[1]=ffn_output_size
        super().__init__(**kwargs)
        self.attention = Multihead_Attention(
            num_heads,dropout,query_size,key_size,value_size,hidden_size,knowledge_dim=1023)
        self.addnorm1 = AddNorm(normshape,dropout)
        self.ffn = PositionWiseFFN(hidden_size,ffn_hidden_size,ffn_output_size,dropout)
        self.addnorm2 = AddNorm(normshape,dropout)

    def forward(self,X):
        Y = self.addnorm1(X,self.attention(X,X,X))
        return self.addnorm2(Y,self.ffn(Y))


class Encoder(nn.Module):

    def __init__(self,num_encoder,embeding_size,patch_size,img_size,num_channel,
                 num_heads,dropout,
                 query_size,key_size,value_size,hidden_size,
                 normshape,ffn_hidden_size,
                 ffn_output_size,**kwargs):
        super().__init__(**kwargs)
        self.embedder = PatchEmbedding(embeding_size,patch_size,img_size,num_channel)
        self.blks = nn.Sequential()
        for i in range(num_encoder):
            self.blks.add_module(f'encoder_block{i}',
                                 Encoder_Block(num_heads,dropout,query_size,
                                               key_size,value_size,hidden_size,
                                               normshape,ffn_hidden_size,ffn_output_size))
            
    def forward(self,X):
        X = self.embedder(X)
        for blk in self.blks:
            X = blk(X)
        return X

class clsdecoder(nn.Module):
    def __init__(self,num_feature,hidden_size,dropout,patch_size):
        super().__init__()
        self.dense1 = nn.Linear(num_feature,hidden_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.patch_size = patch_size
        self.dense2 = nn.Linear(hidden_size,2)

    def forward(self,X):
        # batch_Size, num_patch, embedding_size
        X = torch.mean(X,dim=1)
        X = self.relu(self.dropout(self.dense1(X)))
        # X = nn.Softmax(dim=-1)(self.dense2(X[:,0,:]))
        X = self.dense2(X)
        return X


    

class ViT(nn.Module):
    def __init__(self,num_encoder,embedding_size,num_heads,normshape,ffn_hidden_size,
                 dec_hidden_size,patch_size:int=32,img_size:int=512,num_channel:int=3,dropout:int=0.1,**kwargs):
        super().__init__()
        self.encoder = Encoder(num_encoder,embedding_size,patch_size,img_size,num_channel,
                 num_heads,dropout,embedding_size,embedding_size,embedding_size,embedding_size,
                 normshape,ffn_hidden_size,embedding_size)
        self.decoder = clsdecoder(embedding_size,dec_hidden_size,dropout,patch_size)

    def forward(self,X):
        return self.decoder(self.encoder(X))
        

# x = torch.randn([2,3,512,512],dtype=torch.float32)


# vit = ViT(num_encoder=4,embedding_size=128,patch_size=32,img_size=512,num_channel=3,num_heads=8,dropout=0,
#           normshape=[257,128],ffn_hidden_size=256,dec_hidden_size=48)

# print(vit(x).shape)
# vit.to(torch.device('cuda'))
# summary(vit,(3,512,512))

