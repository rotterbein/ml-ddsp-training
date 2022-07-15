import torch
import yaml
from effortless_config import Config
from os import path, makedirs, system
from ddsp.model import DDSPDecoder, MfccDecoder
import soundfile as sf
from preprocess import get_files

torch.set_grad_enabled(False)


class args(Config):
    CONFIG = "config.yaml"


args.parse_args()

with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

architecture = config["train"]["architecture"]

makedirs(path.join(config["export"]["out_dir"], architecture, config["train"]["model_name"]), exist_ok=True)


class ScriptDDSPDecoder(torch.nn.Module):
    def __init__(self, model, mean_loudness, std_loudness):
        super().__init__()
        self.model = model
        self.model.gru.flatten_parameters()

        self.register_buffer("mean_loudness", torch.tensor(mean_loudness))
        self.register_buffer("std_loudness", torch.tensor(std_loudness))

    def forward(self, pitch, loudness):
        loudness = (loudness - self.mean_loudness) / self.std_loudness
        pitch = pitch[:, ::self.model.block_size]
        loudness = loudness[:, ::self.model.block_size]

        if architecture == "audio_decoder":
            return self.model.realtime_forward_audio(pitch, loudness)
        else:
            return self.model.realtime_forward_controls(pitch, loudness)


class ScriptMfccDecoder(torch.nn.Module):
    def __init__(self, model, mean_loudness, std_loudness):
        super().__init__()
        self.model = model
        self.model.gru.flatten_parameters()

        self.register_buffer("mean_loudness", torch.tensor(mean_loudness))
        self.register_buffer("std_loudness", torch.tensor(std_loudness))

    def forward(self, pitch, loudness, mfccs):
        loudness = (loudness - self.mean_loudness) / self.std_loudness
        pitch = pitch[:, ::self.model.block_size]
        loudness = loudness[:, ::self.model.block_size]
        mfccs = mfccs[:, ::self.model.block_size, :]

        return self.model.realtime_forward_controls(pitch, loudness, mfccs)


with open(path.join(config["train"]["out_dir"], architecture, config["train"]["model_name"], "config.yaml"), "r") as out_config:
    out_config = yaml.safe_load(out_config)

if architecture == "latent_decoder":
    decoder = MfccDecoder(hidden_size=config["model"]["hidden_size"],
                          n_harmonic=config["model"]["n_harmonic"],
                          n_bands=config["model"]["n_bands"],
                          mfcc_bins=config["mfcc"]["mfcc_bins"],
                          sampling_rate=config["model"]["sampling_rate"],
                          block_size=config["model"]["block_size"])
else:
    decoder = DDSPDecoder(**out_config["model"])

state = decoder.state_dict()
pretrained = torch.load(path.join(out_config["train"]["out_dir"], architecture, out_config["train"]["model_name"], "decoder_state.pth"), map_location="cpu")
state.update(pretrained)
decoder.load_state_dict(state)

name = path.normpath(out_config["train"]["model_name"])

if architecture == "latent_decoder":
    scripted_model = torch.jit.script(
        ScriptMfccDecoder(
            decoder,
            out_config["data"]["mean_loudness"],
            out_config["data"]["std_loudness"],
        ))
else:
    scripted_model = torch.jit.script(
        ScriptDDSPDecoder(
            decoder,
            out_config["data"]["mean_loudness"],
            out_config["data"]["std_loudness"],
        ))

torch.jit.save(
    scripted_model,
    path.join(out_config["export"]["out_dir"], architecture, out_config["train"]["model_name"], f"{name}_model.ts"),
)

impulse = decoder.reverb.build_impulse().reshape(-1).numpy()
sf.write(
    path.join(out_config["export"]["out_dir"], architecture, out_config["train"]["model_name"], f"{name}_impulse.wav"),
    impulse,
    out_config["preprocess"]["sampling_rate"],
)

with open(
        path.join(out_config["export"]["out_dir"], architecture, out_config["train"]["model_name"], f"{name}_config.yaml"),
        "w",
) as copy_config:
    yaml.safe_dump(out_config, copy_config)
