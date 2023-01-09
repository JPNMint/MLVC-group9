import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from util.vision_transformer_util import bcolors

########### TO-DO ###########
# 1. Implement and utilize positional embedding
#   --> See: self.pos_embedding = nn.Parameter()
#   --> See: x += self.pos_embedding
# 2. Implement and utilize attention mechanism
#   --> See: TODO: Implement attention mechanism
#   --> See: TODO: Use attention mechanism

# helpers


def pair(t):
    """Check if input is a pair and return a pair if not.
    
    Args:
        t (object) or (object, object): input
    Returns:
        Tuple of object: (object, object)
    """
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    """Pre-Normalization Module"""

    def __init__(self, dim, fn):
        """Initialize Pre-Normalization module

        Args:
            dim (int): Dimension of the fully connected layer in the transformer
            fn (nn.Module): Function (Either Attention or FeedForward)
        
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """Run input though pre-normalization module
        
        Args:
            x (torch.array): Input batch

        Returns:
            x (torch.array): Output array
        
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Feed Forward Module"""

    def __init__(self, dim, hidden_dim):
        """Initialize Feed Forward Network
        
        Args:
            dim (int): Dimension of the fully connected layer in the transformer
            hidden_dim (int): Dimension of the MLP            
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, dim))

    def forward(self, x):
        """Run input though feed forward network
        
        Args:
            x (torch.array): Input batch

        Returns:
            x (torch.array): Output array
        """
        return self.net(x)


class Attention(nn.Module):
    """Transformer Attention Module"""

    def __init__(self, dim, heads=8, dim_head=64):
        """Initialize Attention Module
        
        Args:
            dim (int): Dimension of the fully connected layer in the transformer
            heads (int): Number of transformer heads
            dim_head (int): Dimension of the head output
            
        """
        super().__init__()
        # TODO: Implement attention mechanism

    def forward(self, x):
        """Run input through attention layer
        
        Args:
            x (torch.array): Input batch

        Returns:
            out (torch.array): Output array
        """

        # TODO: Use attention mechanism

        return out


class Transformer(nn.Module):
    """Transformer Module"""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        """Initialize Transformer
        
        Args:
            dim (int): Dimension of the fully connected layer in the transformer
            depth (int): Depth of the transformer
            heads (int): Number of transformer heads
            dim_head (int): Dimension of the head output
            mlp_dim (int): Dimension of the MLP
            
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads,
                                           dim_head=dim_head)),
                    PreNorm(dim, FeedForward(dim, mlp_dim))
                ]))

    def forward(self, x):
        """Run input through transformer
        
        Args:
            x (torch.array): Input batch

        Returns:
            x (torch.array): Output array
        """
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """Vision Transformer Class"""

    def __init__(self,
                 *,
                 image_size,
                 patch_size,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 channels=3,
                 dim_head=64):
        """Initialize Vision Transformer
        
        Args:
            image_size (int): Input dimension of the input (either 16 or (16, 16))
            patch_size (int): Dimension of the patches extracted from the input (either 4 or (4,4))
            num_classes (int): number of classes to predict (1 for simplicity, as we do not have images with no class)
            dim (int): Dimension of the fully connected layer in the transformer
            depth (int): Depth of the transformer
            heads (float): Number of transformer heads
            mlp_dim (int): Dimension of the MLP
            pool (string): Pooling type
            channels (int): Number of channels in the input image
            dim_head (int): Dimension of the head output
            
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width //
                                                        patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height,
                      p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        # TODO: Implement positional embedding
        self.pos_embedding = nn.Parameter()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim),
                                      nn.Linear(dim, num_classes))

    def forward(self, batch):
        """Run input through vision transformer
        
        Args:
            batch (torch.array): Input batch

        Returns:
            out (torch.array): Output array
        """
        x = self.to_patch_embedding(batch)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # TODO: Use positional embedding. NOTE: The next line (x += self.pos_embedding) requires modification
        x += self.pos_embedding
        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        out = torch.sigmoid(x)
        return out

    def predict(self, dataloader, device):
        """Predict all samples from a dataloader

        Args: 
            dataloader (torch.utils.data.dataloader.DataLoader): PyTorch Dataloader
            device (torch.device or str): "cpu" or "cuda"
        
        Returns:
            dataframe (pandas dataframe): Table of all predictions
            accuracy (float): Percentage of correct predictions
        """

        array_score = []
        acc = 0

        self.eval()

        for _, (data, label) in enumerate(dataloader, 0):

            data = data.to(device)
            label = label.to(device)
            predictions_raw = self.forward(data)

            predictions_raw_numpy = predictions_raw[:,
                                                    0].cpu().detach().numpy()
            predictions = predictions_raw_numpy.copy()
            predictions = predictions > 0.5

            for i_pred, pred in enumerate(predictions):

                if int(predictions[i_pred]) == int(
                        label[i_pred].cpu().numpy()):
                    correct = bcolors.OKGREEN + "✔" + bcolors.ENDC
                else:
                    correct = bcolors.FAIL + "✖" + bcolors.ENDC

                if pred == False:
                    array_score.append([
                        'Circle',
                        int(predictions[i_pred]),
                        round(predictions_raw_numpy[i_pred], 3),
                        int(label[i_pred].cpu().numpy()), correct
                    ])
                elif pred == True:
                    array_score.append([
                        'Square',
                        int(predictions[i_pred]),
                        round(predictions_raw_numpy[i_pred], 3),
                        int(label[i_pred].cpu().numpy()), correct
                    ])

            acc += np.average((label.cpu().numpy() - predictions) == 0)

        dataframe = pd.DataFrame(
            array_score,
            columns=['class', 'Pred', 'RAW_Pred', 'GT', 'Correct?'])

        return dataframe, acc / len(dataloader)
