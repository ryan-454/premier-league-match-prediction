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
    data = pd.read_csv(r'C:\Users\Ryan George\OneDrive\Desktop\study\Prem_pred\proj3\final_data2.csv')
    
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
def get_match_features(data, team, opponent, date, venue, referee, predictors):
    date = pd.Timestamp(date)
     # calculate matchweek from date

    season_data = data[data["date"] <= date]
    if len(season_data) > 0:
        matchweek = int(season_data.iloc[-1]["matchweek"])
    else:
        matchweek = 1

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

CURRENT_TEAMS = sorted(data["team"].unique().tolist())


st.success("Model ready!")

# ── Input form ───────────────────────────────────────────────
st.subheader("Match Details")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", CURRENT_TEAMS, index=CURRENT_TEAMS.index("Arsenal"))
with col2:
    away_teams = [t for t in CURRENT_TEAMS if t != home_team]
    away_team  = st.selectbox("Away Team", away_teams)

match_date = st.date_input("Match Date")


referees = sorted(data["referee"].dropna().unique().tolist())
referee  = st.selectbox("Referee", referees)

# ── Predict button ────────────────────────────────────────────
if st.button("Predict", type="primary"):
    features = get_match_features(
        data, home_team, away_team, match_date, "Home", referee, predictors
    )

    if features is None:
        st.error("Not enough historical data for one or both teams.")
    else:
        proba        = model.predict_proba(features)[0]
        win_prob     = proba[1]
        not_win_prob = proba[0]

        st.session_state.win_prob         = win_prob
        st.session_state.not_win_prob     = not_win_prob
        st.session_state.home_team        = home_team
        st.session_state.away_team        = away_team
        st.session_state.match_date       = match_date
        st.session_state.prediction_ready = True

# ── Show results if prediction exists ────────────────────────
if st.session_state.get("prediction_ready"):
    win_prob     = st.session_state.win_prob
    not_win_prob = st.session_state.not_win_prob
    home_team    = st.session_state.home_team
    away_team    = st.session_state.away_team
    match_date   = st.session_state.match_date

    # ── Prediction ────────────────────────────────────────
    st.subheader("Prediction")
    col_a, col_b = st.columns(2)
    with col_a:
        if win_prob >= 0.5:
            st.success(f"✅ Predicted: **{home_team} WIN**")
        else:
            st.warning(f"❌ Predicted: **{home_team} NOT WIN**")
    with col_b:
        st.metric("Win Probability",     f"{win_prob*100:.1f}%")
        st.metric("Not Win Probability", f"{not_win_prob*100:.1f}%")


    # ── Form stats ────────────────────────────────────────
    st.subheader("Current Form")
    date_ts  = pd.Timestamp(match_date)
    team_row = data[(data["team"]==home_team) & (data["date"] < date_ts)].sort_values("date").iloc[-1]
    opp_row  = data[(data["team"]==away_team) & (data["date"] < date_ts)].sort_values("date").iloc[-1]

    form_df = pd.DataFrame({
        "Stat":    ["League Position", "Points", "Goal Difference", "PPG (last 5)", "Streak", "Clean Sheets (last 5)", "H2H Win Rate"],
        home_team: [
            int(team_row["league_position"]),
            int(team_row["points_before"] + team_row["points"]),
            int(team_row["gd_before"] + team_row["gd"]),
            round(team_row["ppg_last5"], 2),
            int(team_row["streak"]),
            int(team_row["clean_sheets_last5"]),
            round(team_row["h2h_win_rate"], 2),
        ],
        away_team: [
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

   # ── Betting Analysis ──────────────────────────────────
    st.divider()
    st.subheader("🎰 Betting Analysis")
    st.caption("Enter bookmaker odds to see if this match is worth betting on")

    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        home_odds = st.number_input(f"{home_team} Win", min_value=1.01, value=2.10, step=0.05)
    with col_o2:
        draw_odds = st.number_input("Draw", min_value=1.01, value=3.40, step=0.05)
    with col_o3:
        away_odds = st.number_input(f"{away_team} Win", min_value=1.01, value=3.60, step=0.05)

    with st.expander("⚙️ Betting Settings"):
        bankroll = st.number_input("Bankroll (₹)", min_value=100, value=500, step=100)
        omega    = st.slider("Kelly Fraction", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
        min_edge = st.slider("Minimum Edge (%)", min_value=1, max_value=20, value=5) / 100
        max_frac = st.slider("Max Bet % of Bankroll", min_value=1, max_value=20, value=10) / 100

    if st.button("Analyse Bet", type="secondary"):
        raw_sum   = (1/home_odds) + (1/draw_odds) + (1/away_odds)
        margin    = (raw_sum - 1) / raw_sum
        fair_home = (1/home_odds) / raw_sum
        edge_home = win_prob - fair_home

        b      = home_odds - 1
        raw_k  = max((b * win_prob - (1 - win_prob)) / b, 0)
        frac_k = raw_k * omega
        capped = min(frac_k, max_frac)
        amt_h  = round(capped * bankroll, 2)

        st.session_state.edge_home = edge_home

        edge_df = pd.DataFrame({
            "Outcome":        [f"{home_team} Win"],
            "Model Prob":     [f"{win_prob*100:.1f}%"],
            "Fair Prob":      [f"{fair_home*100:.1f}%"],
            "Edge":           [f"{edge_home*100:+.1f}%"],
            "Bet Amount (₹)": [f"₹{amt_h:,}" if edge_home >= min_edge else "SKIP"],
        })
        st.dataframe(edge_df, hide_index=True, use_container_width=True)
        st.caption(f"Bookmaker margin: {margin*100:.1f}%")

        if edge_home >= min_edge and amt_h > 0:
            st.success(f"✅ **Best Bet: {home_team} Win — ₹{amt_h:,}**")
        else:
            st.warning(f"⏭️ **SKIP** — Edge of {edge_home*100:+.1f}% is below your minimum of {min_edge*100:.0f}%")

    # ── Simulation ────────────────────────────────────────────
    edge_home = st.session_state.get("edge_home", 0)

    st.divider()
    st.subheader("📈 Betting Simulation")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        num_bets = st.number_input("Number of Bets", min_value=10, max_value=1000, value=100, step=10)
        bet_size = st.number_input("Bet Size (₹)", min_value=10, max_value=10000, value=100, step=10)
    with col_s2:
        num_sims = st.number_input("Number of Simulations", min_value=100, max_value=2000, value=500, step=100)

    if st.button("Run Simulation", type="secondary"):
        import matplotlib.pyplot as plt

        def run_simulation(win_prob, odds, start_bankroll, num_bets, bet_size, num_sims):
            net_profit = bet_size * (odds - 1)
            ev = (win_prob * net_profit) - ((1 - win_prob) * bet_size)
            all_paths, final_values = [], []
            for _ in range(num_sims):
                history = [start_bankroll]
                for _ in range(num_bets):
                    current = history[-1]
                    if current < bet_size:
                        history.append(current)
                        continue
                    if np.random.random() < win_prob:
                        history.append(current + net_profit)
                    else:
                        history.append(current - bet_size)
                all_paths.append(history)
                final_values.append(history[-1])
            return ev, np.array(all_paths), np.array(final_values)

        ev, all_paths, final_values = run_simulation(
            win_prob, home_odds, bankroll, num_bets, bet_size, num_sims
        )

        p10 = np.percentile(all_paths, 10, axis=0)
        p50 = np.percentile(all_paths, 50, axis=0)
        p90 = np.percentile(all_paths, 90, axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        for path in all_paths:
            ax1.plot(path, alpha=0.03, color='green' if edge_home >= 0 else 'red', linewidth=1)
        ax1.plot(p50, color='green' if edge_home >= 0 else 'red', linewidth=2, label='Median')
        ax1.plot(p10, color='orange', linewidth=2, linestyle='--', label='10th pct')
        ax1.plot(p90, color='blue',   linewidth=2, linestyle='--', label='90th pct')
        ax1.axhline(y=bankroll, color='black', linestyle='--', label='Start')
        ax1.set_title(f"{home_team} Win\nModel: {win_prob*100:.1f}% | Edge: {edge_home*100:+.1f}%")
        ax1.set_xlabel("Bets")
        ax1.set_ylabel("Bankroll (₹)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        pct_profitable = (final_values > bankroll).mean() * 100
        ax2.hist(final_values, bins=40, color='green' if edge_home >= 0 else 'red', alpha=0.7, edgecolor='black')
        ax2.axvline(x=bankroll, color='black', linestyle='--', label='Start')
        ax2.axvline(x=np.median(final_values), color='green', linestyle='-', label=f'Median: ₹{np.median(final_values):,.0f}')
        ax2.axvline(x=np.percentile(final_values, 10), color='orange', linestyle='--', label=f'10th: ₹{np.percentile(final_values,10):,.0f}')
        ax2.set_title(f"Final Bankroll Distribution\n{pct_profitable:.1f}% profitable | EV: ₹{ev:.2f}/bet")
        ax2.set_xlabel("Final Bankroll (₹)")
        ax2.set_ylabel("Frequency")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

    # ── Bankroll tracker & bet history ────────────────────
    st.divider()
    st.subheader("📊 Bankroll Tracker")

    if "bet_history" not in st.session_state:
        st.session_state.bet_history = []
    if "bankroll_current" not in st.session_state:
        st.session_state.bankroll_current = 500.0

    with st.form("record_result"):
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            rec_match = st.text_input("Match", value=f"{home_team} vs {away_team}")
        with col_r2:
            rec_bet   = st.number_input("Bet Amount (₹)", min_value=0.0, value=0.0)
        with col_r3:
            rec_odds  = st.number_input("Odds taken", min_value=1.01, value=2.10, step=0.05)
        rec_won   = st.radio("Result", ["Home Win", "Not Home Win"], horizontal=True)
        submitted = st.form_submit_button("Record Result")

        if submitted and rec_bet > 0:
            won  = rec_won == "Home Win"
            pnl  = round(rec_bet * (rec_odds - 1), 2) if won else -rec_bet
            st.session_state.bankroll_current += pnl
            st.session_state.bet_history.append({
                "Match":    rec_match,
                "Bet (₹)":  rec_bet,
                "Odds":     rec_odds,
                "Result":   "✅ Won" if won else "❌ Lost",
                "P&L (₹)":  f"₹{pnl:+,}",
                "Bankroll": f"₹{st.session_state.bankroll_current:,}"
            })
            st.success(f"Recorded! P&L: ₹{pnl:+,} | Bankroll: ₹{st.session_state.bankroll_current:,}")

    starting = 500.0
    growth   = ((st.session_state.bankroll_current - starting) / starting) * 100
    st.metric("Current Bankroll", f"₹{st.session_state.bankroll_current:,}", f"{growth:+.1f}% from start")

    if st.session_state.bet_history:
        st.subheader("Bet History")
        st.dataframe(pd.DataFrame(st.session_state.bet_history), hide_index=True, use_container_width=True)

#To run it, open your terminal, navigate to the folder and run:
#streamlit run app.py