import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Premier League Match Predictor")


# ── Helper functions ─────────────────────────────────────────
def add_true_league_position(df):
    df = df.sort_values(["season","date","time"]).copy()
    df["league_position"] = np.nan
    for season, season_df in df.groupby("season"):
        teams = season_df["team"].unique()
        table = {team: {"points":0,"gd":0,"gf":0} for team in teams}
        for matchweek, mw_df in season_df.groupby("matchweek"):
            ranked = sorted(table.items(), key=lambda x: (x[1]["points"],x[1]["gd"],x[1]["gf"]), reverse=True)
            pos_map = {team: i+1 for i,(team,_) in enumerate(ranked)}
            df.loc[mw_df.index, "league_position"] = mw_df["team"].map(pos_map)
            for _, row in mw_df.iterrows():
                table[row["team"]]["points"] += row["points"]
                table[row["team"]]["gd"]     += row["gd"]
                table[row["team"]]["gf"]     += row["gf"]
    return df

def compute_form(group, n=5):
    group = group.sort_values("date")
    group[f"wins_last{n}"]   = group["target"].rolling(n, closed="left").sum().fillna(0)
    group[f"draws_last{n}"]  = (group["result"]=="D").astype(int).rolling(n, closed="left").sum().fillna(0)
    group[f"losses_last{n}"] = (group["result"]=="L").astype(int).rolling(n, closed="left").sum().fillna(0)
    return group

def home_away_form(group, n=5):
    group = group.sort_values("date")
    for venue_type, prefix in [("Home","home"),("Away","away")]:
        mask = group["venue"] == venue_type
        group[f"{prefix}_wins_last{n}"] = (
            group["target"].where(mask)
            .rolling(n, min_periods=1, closed="left").sum()
            .where(mask).ffill().fillna(0)
        )
    return group

def compute_streak(group):
    group = group.sort_values("date")
    streaks, current_streak = [], 0
    for result in group["result"].shift(1):
        if pd.isna(result): streaks.append(0)
        elif result == "W":
            current_streak = max(1, current_streak+1) if current_streak >= 0 else 1
            streaks.append(current_streak)
        elif result == "L":
            current_streak = min(-1, current_streak-1) if current_streak <= 0 else -1
            streaks.append(current_streak)
        else:
            current_streak = 0
            streaks.append(0)
    group["streak"] = streaks
    return group

def compute_clean_sheets(group, n=5):
    group = group.sort_values("date")
    group[f"clean_sheets_last{n}"] = (
        (group["ga"]==0).astype(int).rolling(n, closed="left").sum().fillna(0)
    )
    return group

def compute_h2h(data):
    data = data.sort_values(["team","opponent","date"])
    data["h2h_wins"]  = (data.groupby(["team","opponent"])["target"]
                         .apply(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
                         .reset_index(level=[0,1], drop=True).fillna(0))
    data["h2h_games"] = (data.groupby(["team","opponent"])["target"]
                         .apply(lambda x: x.shift(1).rolling(5, min_periods=1).count())
                         .reset_index(level=[0,1], drop=True).fillna(0))
    data["h2h_win_rate"] = (data["h2h_wins"] / data["h2h_games"]).fillna(0)
    return data

def compute_ppg(group, n=5):
    group = group.sort_values("date")
    group[f"ppg_last{n}"] = group["points"].rolling(n, closed="left").mean().fillna(0)
    return group



# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = pd.read_csv('final_data2.csv')
    
    # --- all your preprocessing here ---
    opponent_name_fix = {
        "Manchester Utd": "Manchester United",
        "Newcastle Utd": "Newcastle United",
        "Brighton": "Brighton and Hove Albion",
        "West Brom": "West Bromwich Albion",
        "West Ham": "West Ham United",
        "Wolves": "Wolverhampton Wanderers",
        "Tottenham": "Tottenham Hotspur",
        "Sheffield Utd": "Sheffield United",
        "Nott'ham Forest": "Nottingham Forest",
        "Huddersfield": "Huddersfield Town",
    }
    data["opponent"] = data["opponent"].replace(opponent_name_fix)



    data["date"]       = pd.to_datetime(data["date"], format="%d-%m-%Y")
    data["target"]     = (data["result"] == "W").astype(int)
    data["venue_code"] = data["venue"].astype("category").cat.codes
    data["opp_code"]   = data["opponent"].astype("category").cat.codes
    data["day_code"]   = data["date"].dt.dayofweek
    data["matchweek"]  = data["round"].str.extract(r"(\d+)").astype(int)

    data = data.sort_values(["team", "season", "date"])
    data["gd"]     = data["gf"] - data["ga"]
    data["points"] = 0
    data.loc[data["result"] == "W", "points"] = 3
    data.loc[data["result"] == "D", "points"] = 1

    # league position
    data = add_true_league_position(data)

    # cumulative stats
    data = data.sort_values(["team", "season", "date"]).reset_index(drop=True)
    for col, src in [("gd_before","gd"), ("points_before","points"), ("gf_before","gf"), ("ga_before","ga")]:
        data[col] = data.groupby(["team","season"])[src].cumsum().shift(1).fillna(0).astype(int)

    # form features
    data = (data.reset_index(drop=True).set_index(["team","season"])
            .groupby(["team","season"], group_keys=False).apply(compute_form).reset_index())
    data = (data.reset_index(drop=True).set_index(["team","season"])
            .groupby(["team","season"], group_keys=False).apply(home_away_form).reset_index())
    data = data.sort_values(["team","date"])
    data["days_since_last"] = data.groupby("team")["date"].diff().dt.days.fillna(7).astype(int)
    data = (data.reset_index(drop=True).set_index(["team","season"])
            .groupby(["team","season"], group_keys=False).apply(compute_streak).reset_index())
    data = (data.reset_index(drop=True).set_index(["team","season"])
            .groupby(["team","season"], group_keys=False).apply(compute_clean_sheets).reset_index())
    data = compute_h2h(data)
    data["referee_code"] = data["referee"].astype("category").cat.codes
    data = (data.reset_index(drop=True).set_index(["team","season"])
            .groupby(["team","season"], group_keys=False).apply(compute_ppg).reset_index())

 

    # opponent merge
    data = data.sort_values(["team","season","date"]).reset_index(drop=True)
    opp_form = data[[
        "date","team","wins_last5","draws_last5","losses_last5",
        "home_wins_last5","away_wins_last5","days_since_last","streak",
        "clean_sheets_last5","h2h_wins","h2h_win_rate","ppg_last5",
        "league_position","points_before","gd_before","gf_before","ga_before",
    ]].copy()
    opp_form.columns = [
        "date","opponent","opp_wins_last5","opp_draws_last5","opp_losses_last5",
        "opp_home_wins_last5","opp_away_wins_last5","opp_days_since_last","opp_streak",
        "opp_clean_sheets_last5","opp_h2h_wins","opp_h2h_win_rate","opp_ppg_last5",
        "opp_league_position","opp_points_before","opp_gd_before","opp_gf_before","opp_ga_before",
    ]
    data = data.merge(opp_form, on=["date","opponent"], how="left")
    data["points_diff"]   = data["points_before"] - data["opp_points_before"]
    data["gd_diff"]       = data["gd_before"] - data["opp_gd_before"]
    data["position_diff"] = data["league_position"] - data["opp_league_position"]


    return data




# ── Train model ──────────────────────────────────────────────
@st.cache_resource
def train_model(data):
    predictors = [
        "venue_code","opp_code","day_code","matchweek",
        "gd_before","points_before","league_position","gf_before","ga_before",
        "home_wins_last5","days_since_last","streak","clean_sheets_last5",
        "h2h_win_rate","referee_code","ppg_last5",
        "opp_days_since_last","opp_streak","opp_clean_sheets_last5",
        "opp_h2h_win_rate","opp_ppg_last5","opp_league_position",
        "opp_points_before","opp_gd_before","opp_gf_before","opp_ga_before",
        "points_diff","gd_diff","position_diff"
    ]
    ensemble = VotingClassifier(
        estimators=[
            ("rf",   RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)),
            ("xgb",  XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=1, eval_metric="logloss", verbosity=0)),
            ("gb",   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=1)),
            ("lgbm", LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=1, verbose=-1))
        ],
        voting="soft",
        weights=[19, 16, 1, 6]
    )
    train = data[data["season"].isin([2020,2021,2022,2023,2024,2025])].copy()
    train[predictors] = train[predictors].fillna(0)
    ensemble.fit(train[predictors], train["target"])
    return ensemble, predictors

# ── Get features for a match ─────────────────────────────────
def get_match_features(data, team, opponent, date, venue, matchweek, referee, predictors):
    date = pd.Timestamp(date)
    team_data = data[(data["team"]==team) & (data["date"] < date)].sort_values("date")
    opp_data  = data[(data["team"]==opponent) & (data["date"] < date)].sort_values("date")

    if len(team_data) == 0 or len(opp_data) == 0:
        return None

    t = team_data.iloc[-1]
    o = opp_data.iloc[-1]

    features = {
        "venue_code":               1 if venue=="Home" else 0,
        "opp_code":                 data[data["opponent"]==opponent]["opp_code"].iloc[0],
        "day_code":                 date.dayofweek,
        "matchweek":                matchweek,
        "gd_before":                int(t["gd_before"] + t["gd"]),
        "points_before":            int(t["points_before"] + t["points"]),
        "league_position":          t["league_position"],
        "gf_before":                int(t["gf_before"] + t["gf"]),
        "ga_before":                int(t["ga_before"] + t["ga"]),
        "home_wins_last5":          t["home_wins_last5"],
        "days_since_last":          (date - t["date"]).days,
        "streak":                   t["streak"],
        "clean_sheets_last5":       t["clean_sheets_last5"],
        "h2h_win_rate":             t["h2h_win_rate"],
        "referee_code":             data[data["referee"]==referee]["referee_code"].iloc[0] if referee in data["referee"].values else -1,
        "ppg_last5":                t["ppg_last5"],
        "opp_days_since_last":      (date - o["date"]).days,
        "opp_streak":               o["streak"],
        "opp_clean_sheets_last5":   o["clean_sheets_last5"],
        "opp_h2h_win_rate":         o["h2h_win_rate"],
        "opp_ppg_last5":            o["ppg_last5"],
        "opp_league_position":      o["league_position"],
        "opp_points_before":        int(o["points_before"] + o["points"]),
        "opp_gd_before":            int(o["gd_before"] + o["gd"]),
        "opp_gf_before":            int(o["gf_before"] + o["gf"]),
        "opp_ga_before":            int(o["ga_before"] + o["ga"]),
        "points_diff":              int((t["points_before"]+t["points"]) - (o["points_before"]+o["points"])),
        "gd_diff":                  int((t["gd_before"]+t["gd"]) - (o["gd_before"]+o["gd"])),
        "position_diff":            t["league_position"] - o["league_position"],
       
    }
    return pd.DataFrame([features])[predictors]

# ── Current season teams ──────────────────────────────────────

# ── Main app ─────────────────────────────────────────────────
with st.spinner("Loading data and training model..."):
    data = load_data()
    model, predictors = train_model(data)

CURRENT_TEAMS = sorted(data[data["season"] == 2025]["team"].unique().tolist())


st.success("Model ready!")

# ── Input form ───────────────────────────────────────────────
st.subheader("Match Details")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", CURRENT_TEAMS, index=CURRENT_TEAMS.index("Arsenal"))
with col2:
    away_teams = [t for t in CURRENT_TEAMS if t != home_team]
    away_team  = st.selectbox("Away Team", away_teams)

col3, col4 = st.columns(2)
with col3:
    match_date = st.date_input("Match Date")
with col4:
    matchweek  = st.number_input("Matchweek", min_value=1, max_value=38, value=20)

referees = sorted(data["referee"].dropna().unique().tolist())
referee  = st.selectbox("Referee", referees)

# ── Predict ───────────────────────────────────────────────────
if st.button("Predict", type="primary"):
    features = get_match_features(
        data, home_team, away_team, match_date,
        "Home",  matchweek, referee, predictors
    )

    if features is None:
        st.error("Not enough historical data for one or both teams.")
    else:
        proba = model.predict_proba(features)[0]
        win_prob     = proba[1]
        not_win_prob = proba[0]

        st.subheader("Prediction")
        col_a, col_b = st.columns(2)
        with col_a:
            if win_prob >= 0.5:
                st.success(f"✅ **{home_team} WIN**")
            else:
                st.warning(f"❌ **{home_team} NOT WIN**")
        with col_b:
            st.metric("Win Probability",    f"{win_prob*100:.1f}%")
            st.metric("Not Win Probability", f"{not_win_prob*100:.1f}%")

        # ── Form stats ───────────────────────────────────────
        st.subheader("Current Form")
        date_ts  = pd.Timestamp(match_date)
        team_row = data[(data["team"]==home_team) & (data["date"] < date_ts)].sort_values("date").iloc[-1]
        opp_row  = data[(data["team"]==away_team)  & (data["date"] < date_ts)].sort_values("date").iloc[-1]

        form_df = pd.DataFrame({
            "Stat":         ["League Position", "Points", "Goal Difference", "PPG (last 5)", "Streak", "Clean Sheets (last 5)", "H2H Win Rate"],
            home_team:      [
                int(team_row["league_position"]),
                int(team_row["points_before"] + team_row["points"]),
                int(team_row["gd_before"] + team_row["gd"]),
                round(team_row["ppg_last5"], 2),
                int(team_row["streak"]),
                int(team_row["clean_sheets_last5"]),
                round(team_row["h2h_win_rate"], 2),
            ],
            away_team:      [
                int(opp_row["league_position"]),
                int(opp_row["points_before"] + opp_row["points"]),
                int(opp_row["gd_before"] + opp_row["gd"]),
                round(opp_row["ppg_last5"], 2),
                int(opp_row["streak"]),
                int(opp_row["clean_sheets_last5"]),
                round(opp_row["h2h_win_rate"], 2),
            ],
        })
        st.dataframe(form_df, hide_index=True, use_container_width=True)

#To run it, open your terminal, navigate to the folder and run:
#streamlit run app.py
