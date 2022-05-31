# DDSP Training using PyTorch

Fork of ACIDS-IRCAM's implementation of the [DDSP model](https://github.com/magenta/ddsp) using PyTorch. This implementation can be exported to a torchscript model, which can be used in a real-time environment.


## Usage

Edit the `config.yaml` file to fit your needs (audio location, preprocess folder, sampling rate, model parameters...). The `block_size` must be a power of 2 in order to use the model in a real-time environment. 

Preprocess your data using 

```bash
python preprocess.py [--config <path-to-config-file>]
```

You can then train your model using 

```bash
python train.py [--config <path-to-config-file>]
```

Once trained, export it using

```bash
python export.py [--config <path-to-config-file>]
```
