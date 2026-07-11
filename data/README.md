# Data — how to get it (nothing here is committed)

The competition data is **not** stored in git (see `.gitignore`). Redistributing Kaggle
competition data usually violates the rules — download it yourself.

## 1. One-time: Kaggle API token

1. Kaggle → your avatar → **Settings** → **API** → *Create New Token*.
2. This downloads `kaggle.json`. Put it at `~/.kaggle/kaggle.json` and:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

## 2. Accept the competition rules

You must click **Join Competition** / accept rules on the competition page once, or the
CLI download will 403.

## 3. Download

```bash
kaggle competitions download -c rogii-wellbore-geology-prediction -p data/raw
unzip data/raw/rogii-wellbore-geology-prediction.zip -d data/raw
```

## Expected layout after unzip

```
data/
├── raw/
│   ├── train/
│   │   ├── {WELL}__horizontal_well.csv   # MD, X, Y, Z, GR, TVT (target), TVT_input, formation depths
│   │   ├── {WELL}__typewell.csv          # TVT, GR, Geology (formation label)
│   │   └── {WELL}.png                    # well-path + cross-section image
│   ├── test/                             # ~200 wells; TVT hidden (NaN) in eval zone
│   └── sample_submission.csv             # id = {WELLNAME}_{row_index}, tvt = prediction
└── processed/                            # your cached joins/features (created by src/data.py)
```

> ⚠️ The `test/` folder you see on Kaggle contains only a few **leaked training wells** as
> examples. On submission rerun they are replaced by the real hidden test set.
