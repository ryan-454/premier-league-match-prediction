# Premier League Match Win Predictor

Predicts whether a Premier League team will **win** a given home match (binary: Win vs Not Win) using a weighted soft-voting ensemble trained on historical match data from 2017–2026.

> ⚠️ **Note**: The Streamlit app is a work in progress and may not run correctly out of the box without the required dataset. The notebook pipeline is fully functional.

**Live Demo**: [premier-league-match-prediction.streamlit.app](https://premier-league-match-prediction-6w5bqypkpykcjc4p5pilab.streamlit.app/)

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

## How it works

### Features engineered before each match:
- Venue, opponent, day, matchweek encodings
- Cumulative season stats: goals scored/conceded, points, goal difference (all *before* the match)
- True pre-match league position (computed matchweek by matchweek)
- Rolling form: home wins last 5, PPG last 5
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

- **Train**: seasons 2020–2025 (all available data)
- **Evaluation**: train on 2020–2023, test on 2024/25 season

---

## Streamlit App Features
- Select home/away teams from current season squads
- Enter match date and referee
- Get win probability prediction with current form table
- **Betting Analysis**: enter bookmaker odds to calculate edge, Kelly criterion bet sizing and bookmaker margin
- **Monte Carlo Simulation**: visualise expected bankroll paths and final distribution over N simulated bets
- **Bankroll Tracker**: record bet results and track P&L over time

---

## Data Pipeline
- Historical data (2017–2024): scraped from FBref using custom scraper
- 2025–26 season: scraped using [ScraperFC](https://github.com/oseymour/ScraperFC) with weekly incremental updates
- Raw match objects saved as `.pkl` for reuse without rescraping

---

## Requirements

Python 3.9+ recommended.

```bash
pip install pandas numpy scikit-learn xgboost lightgbm streamlit scraperfc
```

---

## Data

Place your dataset at the path specified in `app.py`. The CSV must contain:

`date`, `time`, `round`, `day`, `venue`, `result`, `gf`, `ga`, `opponent`, `xg`, `xga`, `poss`, `referee`, `sh`, `sot`, `crdy`, `crdr`, `season`, `team`

- `date` format: `%d-%m-%Y` (e.g. `16-08-2024`)
- `result`: one of `W`, `D`, `L`

---

## Usage

### Jupyter Notebook
1. Place dataset at the correct path
2. Open `notebook.ipynb` and run all cells
3. Output: precision, accuracy, classification report

### Streamlit App
```bash
streamlit run app.py
```



