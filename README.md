# Premier League Match Win Predictor

A machine learning app that predicts whether a Premier League team will **win** a given match (binary: Win vs Not Win), built on 8 seasons of historical match data.

🟢 **[Live Demo](https://premier-league-match-prediction-6w5bqypkpykcjc4p5pilab.streamlit.app/)**

---

## Results

| Metric | Score |
|---|---|
| Accuracy | **69.21%** |
| Win Precision | **63.18%** |
| Not Win Recall | 0.84 |
| Win Recall | 0.44 |

> For context, a naive "always predict not-win" baseline achieves ~62% accuracy. Professional football prediction models typically sit in the 55–65% range — this model outperforms that benchmark.

---

## How It Works

### Model
A weighted soft-voting ensemble of four ML models trained on seasons 2020–2023 and tested on the 2024/25 season:

| Model | Weight |
|---|---|
| Random Forest | 19 |
| XGBoost | 16 |
| LightGBM | 6 |
| Gradient Boosting | 1 |

### Features Engineered (40+)
All features are computed strictly *before* each match to prevent data leakage:

- **Form:** rolling wins last 5, PPG last 5, win/loss streak, clean sheets last 5
- **Season stats:** cumulative goals scored/conceded, points, goal difference
- **Position:** true pre-match league position computed matchweek by matchweek
- **Head-to-head:** win rate across last 5 meetings vs the opponent
- **Match context:** venue, referee encoding, rest days, hour, day, matchweek
- **Opponent mirrors:** all of the above mirrored for the opposing team
- **Difference features:** points diff, goal difference diff, position diff

### Data
6,600+ match records scraped from FBref across 8 seasons (2017–2024) using ScraperFC. The pipeline is designed to incrementally update as new match results are played.

---

## Streamlit App

Select home/away teams, matchweek, date, and referee to get a win probability and current form stats for both teams.

```bash
streamlit run app.py
```

Or try the **[live version](https://premier-league-match-prediction-6w5bqypkpykcjc4p5pilab.streamlit.app/)** instantly.

---

## Run Locally

**Requirements:** Python 3.9+

```bash
pip install pandas numpy scikit-learn xgboost lightgbm streamlit
```

Place your dataset at `final_df2.csv` with the following columns:

`date`, `time`, `round`, `day`, `venue`, `result`, `gf`, `ga`, `opponent`, `xg`, `xga`, `poss`, `referee`, `sh`, `sot`, `crdy`, `crdr`, `season`, `team`

- `date` format: `%d-%m-%Y` (e.g. `16-08-2024`)
- `result`: one of `W`, `D`, `L`

### Jupyter Notebook
1. Place dataset at `final_df2.csv`
2. Open `notebook.ipynb` and run all cells
3. Output: precision, accuracy, classification report

---

## Project Structure

```
├── notebook.ipynb       # full pipeline: feature engineering, training, evaluation
├── app.py               # Streamlit prediction interface
├── data/
│   └── match_df.csv     # match dataset (not included)
└── README.md
```

---

## Tech Stack

`Python` `scikit-learn` `XGBoost` `LightGBM` `Streamlit` `Pandas` `ScraperFC`
