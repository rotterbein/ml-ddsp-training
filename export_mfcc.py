import torch
import yaml
from effortless_config import Config
from os import path, makedirs, system
from ddsp.model import MfccDecoder
import soundfile as sf
from preprocess import get_files

torch.set_grad_enabled(False)


class args(Config):
    CONFIG = "config.yaml"
    REALTIME = True


args.parse_args()

with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

makedirs(path.join(config["export"]["out_dir"], config["train"]["model_name"]), exist_ok=True)


class ScriptDDSP(torch.nn.Module):
    def __init__(self, ddsp, mean_loudness, std_loudness, realtime):
        super().__init__()
        self.ddsp = ddsp
        self.ddsp.gru.flatten_parameters()

        self.register_buffer("mean_loudness", torch.tensor(mean_loudness))
        self.register_buffer("std_loudness", torch.tensor(std_loudness))
        self.realtime = realtime

    def forward(self, pitch, loudness, mfccs):
        loudness = (loudness - self.mean_loudness) / self.std_loudness
        pitch = pitch[:, ::self.ddsp.block_size]
        loudness = loudness[:, ::self.ddsp.block_size]
        mfccs = mfccs[:, ::self.ddsp.block_size, :]
        # return self.ddsp.realtime_forward(pitch, loudness)
        return self.ddsp.realtime_forward_controls(pitch, loudness, mfccs)
        
        '''if self.realtime:
            pitch = pitch[:, ::self.ddsp.block_size]
            loudness = loudness[:, ::self.ddsp.block_size]
            return self.ddsp.realtime_forward(pitch, loudness)
        else:
            return self.ddsp(pitch, loudness)'''


with open(path.join(config["train"]["out_dir"], config["train"]["model_name"], "config.yaml"), "r") as out_config:
    out_config = yaml.safe_load(out_config)

ddsp = MfccDecoder(hidden_size=512, n_harmonic=100, n_bands=65, mfcc_bins=30, sampling_rate=48000, block_size=512)

state = ddsp.state_dict()
pretrained = torch.load(path.join(out_config["train"]["out_dir"], out_config["train"]["model_name"], "decoder_state.pth"), map_location="cpu")
state.update(pretrained)
ddsp.load_state_dict(state)

name = path.normpath(out_config["train"]["model_name"])

scripted_model = torch.jit.script(
    ScriptDDSP(
        ddsp,
        out_config["data"]["mean_loudness"],
        out_config["data"]["std_loudness"],
        args.REALTIME,
    ))
torch.jit.save(
    scripted_model,
    path.join(out_config["export"]["out_dir"], out_config["train"]["model_name"], f"{name}_model.ts"),
)

impulse = ddsp.reverb.build_impulse().reshape(-1).numpy()
sf.write(
    path.join(out_config["export"]["out_dir"], out_config["train"]["model_name"], f"{name}_impulse.wav"),
    impulse,
    out_config["preprocess"]["sampling_rate"],
)

with open(
        path.join(out_config["export"]["out_dir"], out_config["train"]["model_name"], f"{name}_config.yaml"),
        "w",
) as copy_config:
    yaml.safe_dump(out_config, copy_config)
