import torch as th
from pathlib import Path

from config import testArgs, splitArgs

class TryModel:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        assert self.mode in ['test', 'split'], "Mode must be either 'test' or 'split'"
        self.args = testArgs if self.mode == 'test' else splitArgs
    
    def test(self):
        if self.mode != 'test':
            raise NotImplementedError("Test mode not implemented yet") 
        from src.model.backbone import UNet
        net = UNet(in_channels = self.args.n_channels)
        random_input = th.randn(self.args.input_shape)
        return net.forward(random_input)
    
    def split(self):
        if self.mode != 'split':
            raise NotImplementedError("Split mode not implemented yet") 
        import librosa
        import soundfile as sf

        from src.model.isolator import Isolator

        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        isolator = Isolator.load_model(self.args.model_path).to(device).eval()

        fpath_src = Path(self.args.input_path)

        input_wav, _ = librosa.load(
                                        fpath_src, 
                                        sr = self.args.sampling_rate, 
                                        mono = False,
                                        res_type='kaiser_fast',
                                        mono = False,
                                        offset=self.args.offset,
                                        duration=self.args.duration
                                    )
        
        wav = th.Tensor(input_wav).to(device)

        with th.no_grad():
            stems = isolator.separate(wav)

        if self.args.write_src:
            stems["input"] = wav
        for name, stem in stems.items():
            fpath_dst = Path(self.args.output_path) / f"{fpath_src.stem}_{name}.wav"
            print(f"Writing {fpath_dst}")
            fpath_dst.parent.mkdir(exist_ok=True)
            sf.write(fpath_dst, stem.cpu().detach().numpy().T, self.args.sampling_rate, "PCM_16")