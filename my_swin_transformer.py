import torch.nn as nn
import torch.nn.functional as F


class Path_embed(nn.Module): #path剪切
    def __init__(self,patch_size=5, in_channels=3, embed_dim=96):
        """
        :param patch_size:  token
        :param in_channels:
        :param embed_dim: out channels
        """
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)
    def forward(self,x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        return x
class Multi_Head_Self_Attention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads #每个头的维度
        self.qkv = nn.Linear(embed_dim,embed_dim*3,bias = False)
        self.proj = nn.Linear(embed_dim,embed_dim)
    def forward(self,x):
        batch_size,seq_len,embed_dim = x.shape #seq_len = out_channels
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size,seq_len,3,self.num_heads,self.head_dim)
        q,k,v = qkv.permute(2,0,3,1,4) #3 b heads s dim
        #得分
        attn_scores = (q @ k.transpose(-2,-1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_scores,dim=-1)
        attn_output = attn_weights @ v
        #合并头
        attn_output = attn_output.permute(0,2,1,3).reshape(batch_size,seq_len,embed_dim)
        return  self.proj(attn_output)
class Swin_Transformer_Block(nn.Module):
    def __init__(self,embed_dim,num_heads,mlp_ratio = 4.0):
        """
        :param embed_dim:
        :param num_heads:
        :param mlp_ratio: MLP层扩展比率 （表示隐藏层大小是 embed_dim 的几倍）
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Multi_Head_Self_Attention(embed_dim,num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim*mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,embed_dim)
        )
    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Patch_Merging(nn.Module):
    """
    合并减少计算
    """
    def __init__(self,embed_dim):
        super().__init__()
        self.reduction = nn.Linear(embed_dim * 4,embed_dim * 2,bias = False)
    def forward(self,x):
        batch_size, seq_len, embed_dim = x.shape
        x = x.reshape(batch_size, seq_len // 4, 4 * embed_dim)
        x = self.reduction(x)
        return  x
class Swin_Tiny(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.patch_embed = Path_embed()
        self.block1 = nn.Sequential(
            Swin_Transformer_Block(embed_dim=96, num_heads=3),
        )
        self.block2 = nn.Sequential(
            Swin_Transformer_Block(embed_dim=192, num_heads=6),
        )

        self.merge1 = Patch_Merging(embed_dim=96)
        self.merge2 = Patch_Merging(embed_dim=192)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.function = nn.Linear(192,num_classes)
    def forward(self,x):
        x = self.patch_embed(x)
        x = self.block1(x)
        x = self.merge1(x)
    #    x = self.block2(x)
    #    x = self.merge2(x)
        x = self.pool(x.transpose(1,2)).squeeze(-1)
        return self.function(x)
