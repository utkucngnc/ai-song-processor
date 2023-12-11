import torch as th
import torch.nn as nn

class STFT_PARAMS:
    WINDOW_SIZE = 4096
    HOP_SIZE = 1024
    WINDOW = nn.Parameter(th.hann_window(WINDOW_SIZE), requires_grad=False)
    T = 512
    F = 1024
  
class trainArgs:
    dataset = ".data/train/musdb18hq"
    output_dir = None
    fp16 = False
    cpu = True
    max_steps = 100
    num_train_epochs = 1
    per_device_train_batch_size = 1
    effective_batch_size = 4
    max_grad_norm = 0.0
  
class testArgs:
    n_channels = 2
    batch_size = 5
    input_shape = (batch_size, n_channels, 128, 512) # B x C x F x T

class splitArgs:
    sampling_rate: int = 44100
    model_path: str = ".models/2stem/model"
    input_path: str = ".data/input/example.mp3"
    output_path: str = ".data/output/example"
    offset: float = 0.0
    duration: float = 30.0
    write_src: bool = False

