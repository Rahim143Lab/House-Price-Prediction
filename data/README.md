# Data

## Dataset used

This project uses the **Ames Housing dataset** (De Cock, 2011) — 2,930
residential property sales recorded in Ames, Iowa between 2006 and 2010,
with 80+ recorded property characteristics. It is a well-known, freely
available alternative to the older Boston Housing dataset, widely used for
regression teaching and benchmarking.

- Paper: De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing
  Data as an End of Semester Regression Project.* Journal of Statistics
  Education, 19(3).
- Original source: https://jse.amstat.org/v19n3/decock/AmesHousing.txt

## File

```
data/raw/AmesHousing.csv   # 2,930 rows x 82 columns, ~960 KB
```

The file is small enough that it is committed directly to this repository
for reproducibility — no separate download step is required to run the
project.

## Regenerating processed splits

`data/processed/` is intentionally left out of version control. Running

```bash
python -m src.train
```

(or `python -m src.data_loader` on its own) will regenerate
`data/processed/train.csv`, `val.csv`, and `test.csv` using a fixed random
seed (`RANDOM_STATE = 42` in `src/config.py`), so the split is always
reproducible from the raw file.

## If the raw file is ever missing

If `data/raw/AmesHousing.csv` is not present, `src/data_loader.py` will
raise a clear `FileNotFoundError` pointing back to this file rather than
failing silently or fabricating results. To restore it, download the CSV
from a mirror of the dataset (e.g. the GitHub repository
`wblakecannon/ames`, path `data/housing.csv`) and place it at
`data/raw/AmesHousing.csv`.
