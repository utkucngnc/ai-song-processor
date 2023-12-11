from typing import Tuple
import torch as th
import torch.nn as nn
from torch.nn import functional as F

'''
Takes audio spectrogram as input and outputs a spectrogram of the same shape
'''

# Define padding pattern
l, r, t, b = 1, 1, 2, 2

class Encoder(nn.Module):
    def __init__(
                self, 
                in_channels: int, 
                out_channels: int, 
                kernel_size: int = 5, 
                stride: Tuple[int,int] = (2,2),
                eps: float = 1e-3,
                momentum: float = 1e-2
                ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(
                                    num_features = out_channels,
                                    track_running_stats = True,
                                    eps = eps,
                                    momentum = momentum
                                )
        self.relu = nn.LeakyReLU(negative_slope = 0.2)
    
    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        down = self.conv(F.pad(x, (l, r, b, t), mode = 'constant', value = 0))
        return down, self.relu(self.bn(down))

class Decoder(nn.Module):
    def __init__(
                self, 
                in_channels: int, 
                out_channels: int, 
                kernel_size: int = 5, 
                stride: int = 2,
                eps: float = 1e-3,
                momentum: float = 1e-2, 
                dropout: float = 0.0
                ) -> None:
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(
                                    num_features = out_channels,
                                    track_running_stats = True,
                                    eps = eps,
                                    momentum = momentum
                                )
        self.dropout = nn.Dropout2d(p = dropout) if 1 > dropout > 0 else nn.Identity()
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        up = self.trans_conv(x)
        up = up[:, :, l:-r, t:-b]
        return self.dropout(self.relu(self.bn(up)))

class UNet(nn.Module):
    def __init__(
                self,
                in_channels: int,
                num_layers: int,
                ) -> None:
        super().__init__()

        # Down Layers
        down_set = [in_channels] + [2**(i + 4) for i in range(num_layers)]
        self.encoder_layers = nn.ModuleList(
            [
                Encoder(in_channels = in_ch, out_channels = out_ch)
                for in_ch, out_ch in zip(down_set[:-1], down_set[1:])
            ]
        )

        # Up Layers
        up_set = down_set[::-1][:-1]
        up_set.append(1)

        self.decoder_layers = nn.ModuleList(
            [
                Decoder(
                        in_channels = in_ch if i == 0 else in_ch * 2, 
                        out_channels = out_ch,
                        dropout = 0.5 if i < 3 else 0.0
                        )
                for i, (in_ch, out_ch) in enumerate(zip(up_set[:-1], up_set[1:]))
            ]
        )

        # Final Mask
        self.final_mask = nn.Conv2d(1, in_channels, kernel_size = 4, dilation = 2, padding = 3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        # Down
        down_pre_activation = []
        for down in self.encoder_layers:
            conv, x = down(x)
            down_pre_activation.append(conv)
        
        # Up
        for i, up in enumerate(self.decoder_layers):
            if i == 0:
                x = up(down_pre_activation.pop())
            else:
                # Merge Skip Connection
                x = up(th.concat([down_pre_activation.pop(), x], axis = 1))
        
        # Final Mask
        mask = self.sigmoid(self.final_mask(x))
        return mask * x