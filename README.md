# Premier League Match Win Predictor

Predicts whether a Premier League team will **win** a given match (binary: Win vs Not Win) using a weighted soft-voting ensemble of four ML models trained on historical match data from 2017–2024.

**Best results: 0.6318 precision, 0.6921 accuracy** on the 2024/25 season test set.

---

## How it works

### Features engineered before each match:
- Venue, opponent, hour, day, matchweek encodings
- Cumulative season stats: goals scored/conceded, points, goal difference (all *before* the match)
- True pre-match league position (computed matchweek by matchweek)
- Rolling form: wins last 5, home wins last 5, PPG last 5
- Win/loss streak, clean sheets last 5
- Head-to-head win rate (last 5 meetings vs this opponent)
- Rest days since last match
- Referee encoding
- All of the above mirrored for the opponent
- Difference features: points diff, GD diff, position diff

### Model:
A `VotingClassifier(voting="soft")` combining:

| Model | Weight |
|---|---|
| Random Forest | 19 |
| XGBoost | 16 |
| Gradient Boosting | 1 |
| LightGBM | 6 |

- **Train**: seasons 2020–2023
- **Test**: matches from 2024-08-16 onward (2024/25 season)

---

## Requirements

Python 3.9+ recommended.
```bash
pip install pandas numpy scikit-learn xgboost lightgbm streamlit
```

---

## Data

Place your dataset at `data/match_df.csv`. The CSV must contain these columns at minimum:

`date`, `time`, `round`, `day`, `venue`, `result`, `gf`, `ga_x`, `opponent`, `xg_x`, `xga`, `poss_x`, `referee`, `sh_x`, `sot`, `crdy`, `crdr`, `season`, `team`

- `date` format: `%d-%m-%Y` (e.g. `16-08-2024`)
- `result`: one of `W`, `D`, `L`

---

## Usage

### Jupyter Notebook
1. Place dataset at `data/match_df.csv`
2. Open `notebook.ipynb` and run all cells
3. Output: precision, accuracy, classification report

### Streamlit App
```bash
streamlit run app.py
```
Opens in browser. Select home/away teams, matchweek, date, kickoff time and referee to get a prediction with win probability and current form stats for both teams.

---

## Results

| Metric | Score |
|---|---|
| Precision (Win) | 0.6318 |
| Accuracy | 0.6921 |
| Not Win recall | 0.84 |
| Win recall | 0.44 |

For context, a naive "always predict not-win" baseline achieves ~62% accuracy. Professional football prediction models typically sit in the 55–65% accuracy range.

---

## Project structure
```
├── notebook.ipynb       # full pipeline
├── app.py               # Streamlit prediction interface
├── data/
│   └── match_df.csv     # match dataset (not included)
└── README.md
```
