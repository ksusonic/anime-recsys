# Telepokegan

Telepokegan is a Python DL-GAN application via Telegram messenger!

## Installation

Use the package manager [poetry](https://python-poetry.org) to install foobar.

```bash
poetry install
```

## Usage

1. Clone repo: `git clone https://github.com/ksusonic/tele-poke-gan`
2. Set up poetry dependencies: `poetry install`
3. Setup project: `pre-commit install`
4. [Optional] You can check pre-commit hooks: `pre-commit run -a`
5. Train model: `python train.py`
6. Infer model: `python infer.py`


## Checklist

### Main
- [ ] train.py
- [ ] infer.py

### Tools
- [X] Poetry
- [X] Pre-commit
- [X] DVC
- [ ] Hydra
- [ ] Logging (MLFlow)

### Optional
- [ ] `fire` instead of `argparse`
- [ ] yaml configs
- [ ] `pathlib` instead of `os.path`
