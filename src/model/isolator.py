import torch as th
import torch.nn.functional as F
from typing import Tuple, List, Dict
import torch.nn as nn
import math

from src.model.backbone import UNet
from config import STFT_PARAMS

class Isolator(nn.Module):
    def __init__(self, stem_names: List[str]) -> None:
        super(Isolator, self).__init__()
        assert len(stem_names) > 0, "Must provide at least one stem name"

        self.F = STFT_PARAMS.F
        self.T = STFT_PARAMS.T
        self.H = STFT_PARAMS.HOP_SIZE
        self.W = STFT_PARAMS.WINDOW_SIZE
        self.window = STFT_PARAMS.WINDOW

        self.stems = nn.ModuleDict({name: UNet(in_channels=2) for name in stem_names})
    
    def forward(self, wav: th.Tensor) -> th.Tensor:
        """
        Separates stereo wav into different tracks (1 predicted track per stem)
        Args:
            wav (tensor): 2 x L
        Returns:
            masked stfts by track name
        """
        # Compute STFT
        stft, mag = self.computeSTFT(wav.squeeze())

        # 1 x 2 x F x T
        mag = mag.unsqueeze(-1).permute([3, 0, 1, 2])
        mag = self.batchifyAudio(mag, self.T) # B x 2 x F x T
        mag = mag.transpose(2,3) # B x 2 x T x F

        # Compute Stem Mask
        masks = {name: stem(mag) for name, stem in self.stems.items()}

        # Compute Denominator
        mask_sum = sum([m**2 for m in masks.values()]) + 1e-10
        
        return {name: self.applyMask(mask, mask_sum, stft) for name, mask in masks.items()}
    
    def isolate(self, wav: th.Tensor) -> Dict[str, th.Tensor]:
        """
        Separates stereo wav into different tracks (1 predicted track per stem)
        Args:
            wav (tensor): 2 x L
        Returns:
            wavs by track name
        """

        stft_masks = self.forward(wav)

        return {
            name: self.computeInverseSTFT(stft_masked)
            for name, stft_masked in stft_masks.items()
        }
    
    # Compute STFT
    def computeSTFT(
                self,
                audio: th.Tensor
                ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the STFT of a given audio signal.
        :param audio: audio signal
        :param window_size: window size
        :param window_func: window function
        :param hop_size: hop size
        :param f: number of frequencies to keep
        :return: STFT of the audio signal
        """
        window_size = self.W
        window_func = self.window
        hop_size = self.H
        f = self.F

        assert len(window_func.shape) == 1 and window_func.shape[0] <= window_size, \
            "window_func must be a 1D tensor of length smaller than window_size"

        stft = th.stft(
                audio,
                n_fft = window_size,
                hop_length = hop_size,
                window = window_func,
                center = True,
                return_complex = False,
                pad_mode="constant",
            )

        # only keep freqs smaller than base frequency
        stft = stft[:, : f, :, :]
        real = stft[:, :, :, 0]
        im = stft[:, :, :, 1]
        mag = th.sqrt(real**2 + im**2)

        return stft, mag

    def applyMask(self, mask: th.Tensor, masks_sum: float, stft: th.Tensor) -> th.Tensor:
        """
        Apply a mask to a given STFT.
        :param mask: mask to apply
        :param masks_sum: sum of all masks
        :param stft: STFT to apply mask to
        :return: masked STFT
        """
        mask = (mask**2 + 1e-10 / 2) / masks_sum
        mask = mask.transpose(2, 3) # B x 2 x F x T

        mask = th.cat(th.split(mask, 1, dim = 0), dim = 3)
        mask = mask.squeeze(0)[:, :, :stft.size(2)].unsqueeze(-1) # 2 x F x L x 1
        return mask * stft

    # Compute inverse STFT
    def computeInverseSTFT(self, stft: th.Tensor) -> th.Tensor:
        """
        Compute the inverse STFT of a given STFT.
        :param stft: STFT
        :param window_size: window size
        :param window_func: window function
        :param hop_size: hop size
        :param t: number of time steps to keep
        :return: inverse STFT of the STFT
        """
        window_size = self.W
        window_func = self.window
        hop_size = self.H

        assert len(window_func.shape) == 1 and window_func.shape[0] <= window_size, \
            "window_func must be a 1D tensor of length smaller than window_size"

        pad = window_size // 2 + 1 - stft.size(1)
        stft = F.pad(stft, (0, 0, 0, 0, 0, pad))

        return th.istft(
                        stft,
                        n_fft = window_size,
                        hop_length = hop_size,
                        window = window_func,
                        center = True
                    ).detach()

    def batchifyAudio(self, tensor: th.Tensor, T: int) -> th.Tensor:
        """
        Partition tensor into segments of length T, zero pad any ragged samples
        Args:
            tensor(Tensor): BxCxFxL
        Returns:
            tensor of size (B*[L/T] x C x F x T)
        """
        # Zero pad the original tensor to an even multiple of T
        orig_size = tensor.size(-1)
        new_size = math.ceil(orig_size / T) * T
        tensor = F.pad(tensor, [0, new_size - orig_size])
        # Partition the tensor into multiple samples of length T and stack them into a batch
        return th.cat(th.split(tensor, T, dim=-1), dim=0)
    
    @classmethod
    def load_model(cls, model_path: str):
        return cls(stem_names = ['vocals', 'accompaniment']).load_state_dict(th.load(model_path))


        
