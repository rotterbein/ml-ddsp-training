import yaml
import pathlib
import librosa as li
from ddsp.core import extract_loudness, extract_pitch
from effortless_config import Config
import numpy as np
from tqdm import tqdm
import numpy as np
from os import makedirs, path
import torch
from scipy.io import wavfile


def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def preprocess(f, sampling_rate, block_size, signal_length, oneshot, **kwargs):
    x, sr = li.load(f, sampling_rate)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)

    return x, pitch, loudness


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        self.signal = np.load(path.join(out_dir, "signal.npy"))
        self.pitch = np.load(path.join(out_dir, "pitch.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))

    def __len__(self):
        return self.signal.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.signal[idx])
        p = torch.from_numpy(self.pitch[idx])
        l = torch.from_numpy(self.loudness[idx])
        return s, p, l


def main():
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    files = get_files(**config["data"])
    pb = tqdm(files)

    signal = []
    pitch = []
    loudness = []

    for f in pb:
        pb.set_description(str(f))
        x, p, l = preprocess(f, **config["preprocess"])
        signal.append(x)
        pitch.append(p)
        loudness.append(l)

    signal = np.concatenate(signal, 0).astype(np.float32)
    pitch = np.concatenate(pitch, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)

    out_dir = config["preprocess"]["out_dir"]
    makedirs(out_dir, exist_ok=True)

    np.save(path.join(out_dir, "signal.npy"), signal)
    np.save(path.join(out_dir, "pitch.npy"), pitch)
    np.save(path.join(out_dir, "loudness.npy"), loudness)


if __name__ == "__main__":
    main()