# [Anime Recommendation System](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

## Installation

Use the package manager [poetry](https://python-poetry.org):

```bash
poetry install
```

## Usage

1. Clone repo: `git clone https://github.com/ksusonic/anime-recsys`
2. Set up poetry dependencies: `poetry install`
3. Setup project: `pre-commit install`
4. [Optional] You can check pre-commit hooks: `pre-commit run -a`
5. Train model: `python anime-recsys/train.py`
6. Infer model: `python anime-recsys/infer.py`


## Checklist

### Main
- [X] train.py
- [X] infer.py

### Tools
- [X] Poetry
- [X] Pre-commit
- [X] DVC
- [X] Hydra
- [X] Logging (MLFlow)

### In planning:
- [] Telegram bot interaction
