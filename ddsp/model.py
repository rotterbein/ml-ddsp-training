import torch
import torch.nn as nn
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample
from .core import harmonic_synth, amp_to_impulse_response, fft_convolve
from .core import resample
import math
import torchaudio.transforms


class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


class DDSP(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate, block_size):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2)
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, loudness):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        #reverb part
        signal = self.reverb(signal)

        return signal

    def realtime_forward(self, pitch, loudness):  # CPUFloatType{1,4,1}  # 4 = buffer_size / block_size
        hidden = torch.cat([
            self.in_mlps[0](pitch),  # CPUFloatType{1,4,512}
            self.in_mlps[1](loudness),  # CPUFloatType{1,4,512}
        ], -1)  # CPUFloatType{1,4,1024}

        gru_out, cache = self.gru(hidden, self.cache_gru)  # CPUFloatType{1,4,512}
        self.cache_gru.copy_(cache)

        hidden = torch.cat([gru_out, pitch, loudness], -1)  # CPUFloatType{1,4,514}
        hidden = self.out_mlp(hidden)  # CPUFloatType{1,4,512}

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))  # CPUFloatType{1,4,101}, element 0: total amplitude, 1-101: harmonic amps

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]  # CPUFloatType{1,4,100}

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp  # CPUFloatType{1,4,100}

        amplitudes = upsample(amplitudes, self.block_size)  # CPUFloatType{1,1024,100}
        pitch = upsample(pitch, self.block_size)  # CPUFloatType{1,1024,1}

        # real-time adapted harmonic synth
        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / self.sampling_rate, 1)

        omega = omega + self.phase
        self.phase.copy_(omega[0, -1, 0] % (2 * math.pi))

        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)

        harmonic = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)  # CPUFloatType{1,1024,1}

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)  # CPUFloatType{1,4,65}

        impulse = amp_to_impulse_response(param, self.block_size)  # CPUFloatType{1,4,256}
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1  # CPUFloatType{1,4,256}

        noise = fft_convolve(noise, impulse).contiguous()  # CPUFloatType{1,4,256}
        noise = noise.reshape(noise.shape[0], -1, 1)  # CPUFloatType{1,1024,1}

        signal = harmonic + noise

        return signal

    def realtime_forward_controls(self, pitch, loudness):  # shapes if buffer_size = block_size # hidden_size = 512
        hidden = torch.cat([
            self.in_mlps[0](pitch),  # CPUFloatType{1,1,512}
            self.in_mlps[1](loudness),  # CPUFloatType{1,1,512}
        ], -1)  # CPUFloatType{1,4,1024}

        gru_out, cache = self.gru(hidden, self.cache_gru)  # CPUFloatType{1,1,512}
        self.cache_gru.copy_(cache)

        hidden = torch.cat([gru_out, pitch, loudness], -1)  # CPUFloatType{1,1,514}
        hidden = self.out_mlp(hidden)  # CPUFloatType{1,1,512}

        # harmonic part
        param = scale_function(
            self.proj_matrices[0](hidden))  # CPUFloatType{1,1,101}, element 0: total amplitude, 1-101: harmonic amps

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]  # CPUFloatType{1,1,100}

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)  # dividing by zero?
        amplitudes *= total_amp  # CPUFloatType{1,1,100}

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)  # CPUFloatType{1,1,65}
        magnitudes = param

        return amplitudes, magnitudes


class MfccEncoder(nn.Module):
    def __init__(self, fft_sizes, mel_bins, mfcc_bins, sampling_rate, block_size, signal_length):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("signal_length", torch.tensor(signal_length))
        self.fft_sizes = fft_sizes
        self.mel_bins = mel_bins
        self.mfcc_bins = mfcc_bins

        keywords = dict(n_fft=self.fft_sizes, n_mels=self.mel_bins)

        self.mfcc = torchaudio.transforms.MFCC(sample_rate=self.sampling_rate, n_mfcc=self.mfcc_bins, log_mels=True, melkwargs=keywords)
        self.norm_out = nn.LayerNorm([int(signal_length / block_size), mfcc_bins])

    def forward(self, audio):
        mfccs = self.mfcc(audio)
        mfccs = torch.permute(mfccs, [0, 2, 1])
        mfccs = mfccs[:, :int(self.signal_length / self.block_size), :]

        return self.norm_out(mfccs)


class MfccDecoder(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, mfcc_bins, sampling_rate, block_size):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.mfcc_bins = mfcc_bins

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * (mfcc_bins + 2))  # latents = num mfcc_bins + pitch + loudness
        self.gru = gru((mfcc_bins + 2), hidden_size)  # gru(2, hidden_size)?
        self.out_mlp = mlp(hidden_size + (mfcc_bins + 2), hidden_size, 3)  # mlp(hidden_size + 2, hidden_size, 3)?

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, loudness, mfccs):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness)
        ], -1)

        for i, layer in enumerate(self.in_mlps):  # avoid non-literal indexing of ModuleList by using enumeration
            if i > 1:
                hidden = torch.cat([hidden, layer(mfccs[:, :, i:(i+1)])], -1)

        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)

        for i in range(self.mfcc_bins):
            hidden = torch.cat([hidden, mfccs[:, :, i:(i+1)]], -1)

        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        #reverb part
        signal = self.reverb(signal)

        return signal

    def realtime_forward_controls(self, pitch, loudness, mfccs):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness)
        ], -1)

        for i, layer in enumerate(self.in_mlps):
            if i > 1:
                j = i-2
                hidden = torch.cat([hidden, layer(mfccs[:, :, j:(j + 1)])], -1)

        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)

        for i in range(self.mfcc_bins):
            hidden = torch.cat([hidden, mfccs[:, :, i:(i + 1)]], -1)

        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(
            self.proj_matrices[0](hidden))  # CPUFloatType{1,1,101}, element 0: total amplitude, 1-101: harmonic amps

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]  # CPUFloatType{1,1,100}

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)  # dividing by zero?
        amplitudes *= total_amp  # CPUFloatType{1,1,100}

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)  # CPUFloatType{1,1,65}
        magnitudes = param

        return amplitudes, magnitudes
