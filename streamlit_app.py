import streamlit as st
import pandas as pd
import numpy as np
import datetime
import statsapi

# Set up page configurations
st.set_page_config(page_title="MLB Daily Hit Predictor", page_icon="⚾", layout="wide")

# ==========================================
# 1. CORE ALGORITHM (Cached for Performance)
# ==========================================
def logodds(p):
    return np.log(p / (1 - p))

def probability(odds):
    return 1 / (1 + np.exp(-odds))

def predict_daily_hit_probability(player_data, matchup_data, environment_data):
    base_ba = player_data["batting_average"]
    contact_mod = player_data["contact_rate"] - player_data["whiff_rate"]
    
    base_h_per_pa = base_ba * 0.90 + (contact_mod * 0.05)
    current_log_odds = logodds(base_h_per_pa)
    
    # Platoon Split
    pitcher_hand = matchup_data["pitcher_hand"]
    split_ba = player_data["vs_LHP"] if pitcher_hand == "L" else player_data["vs_RHP"]
    current_log_odds += ((split_ba - base_ba) * 2.5)
    
    # Park & Weather
    current_log_odds += np.log(environment_data["park_factor"])
    current_log_odds += (environment_data["temperature"] - 70) * 0.0015
    
    if environment_data["wind_direction"] == "out": current_log_odds += 0.05
    elif environment_data["wind_direction"] == "in": current_log_odds -= 0.05
    
    adjusted_pa_hit_prob = probability(current_log_odds)
    
    projected_pa = matchup_data["projected_pa"]
    if projected_pa < 3:
        return 0.0
        
    prob_zero_hits = (1 - adjusted_pa_hit_prob) ** projected_pa
    return round((1 - prob_zero_hits) * 100, 1)

# Mocked fast lookup data for example stability
def fetch_mocked_advanced_stats(player_id):
    # In practice, swap this with your live database or PyBaseball engine
    np.random.seed(player_id)
    ba = round(np.random.uniform(0.220, 0.310), 3)
    return {
        "batting_average": ba,
        "contact_rate": round(np.random.uniform(0.70, 0.85), 2),
        "whiff_rate": round(np.random.uniform(0.15, 0.30), 2),
        "vs_LHP": round(ba + np.random.uniform(-0.02, 0.03), 3),
        "vs_RHP": round(ba + np.random.uniform(-0.02, 0.03), 3)
    }

# ==========================================
# 2. DATA PROCESSING PIPELINE
# ==========================================
@st.cache_data(ttl=3600) # Caches the data for 1 hour so the UI remains ultra fast
def load_daily_predictions(date_str):
    schedule = statsapi.schedule(date=date_str)
    master_predictions = []

    for game in schedule:
        game_id = game['game_id']
        if game['status'] in ['Postponed', 'Canceled']:
            continue
            
        try:
            game_data = statsapi.get('game', {'gamePk': game_id})
            weather_info = game_data.get('gameData', {}).get('weather', {})
            temp = int(weather_info.get('temp', 70))
            wind_str = weather_info.get('wind', '0 mph').lower()
            
            wind_direction = "neutral"
            if "out" in wind_str: wind_direction = "out"
            elif "in" in wind_str: wind_direction = "in"

            environment = {"park_factor": 1.00, "temperature": temp, "wind_direction": wind_direction}
            lineups = game_data.get('liveData', {}).get('boxscore', {}).get('teams', {})
            
            for team_side, opposing_hand in [('away', 'R'), ('home', 'L')]: # Simplified pitcher hand assignment
                batting_order = lineups.get(team_side, {}).get('battingOrder', [])
                
                for index, player_id in enumerate(batting_order):
                    if index >= 9: continue # Keep top 9 starting players
                    
                    player_info = statsapi.lookup_player(player_id)
                    if not player_info: continue
                    player_name = player_info[0]['fullName']
                    
                    player_stats = fetch_mocked_advanced_stats(player_id)
                    matchup = {"pitcher_hand": opposing_hand, "projected_pa": 5 if index < 4 else 4}
                    
                    prob = predict_daily_hit_probability(player_stats, matchup, environment)
                    
                    master_predictions.append({
                        "Order": index + 1,
                        "Player": player_name,
                        "Team": game[f'{team_side}_name'],
                        "Opp Pitcher Hand": opposing_hand,
                        "Proj PA": matchup["projected_pa"],
                        "Season BA": player_stats["batting_average"],
                        "Hit Probability (%)": prob
                    })
        except Exception:
            continue
            
    return pd.DataFrame(master_predictions)

# ==========================================
# 3. STREAMLIT UI LAYOUT
# ==========================================
st.title("⚾ MLB Daily Hit Probability Dashboard")
st.markdown("This dashboard calculates the real-time probability of starting players recording **at least 1 hit** based on matchup variables.")

# Sidebar Filters
st.sidebar.header("Dashboard Controls")
selected_date = st.sidebar.date_input("Select Games Date", datetime.date.today())
date_formatted = selected_date.strftime("%m/%d/%Y")

# Min Probability Slider
min_prob = st.sidebar.slider("Minimum Hit Probability (%)", 0, 100, 50)

# Load Data
with st.spinner("Fetching live lineups and environmental variables..."):
    df_predictions = load_daily_predictions(date_formatted)

if not df_predictions.empty:
    # Filter operations
    filtered_df = df_predictions[df_predictions["Hit Probability (%)"] >= min_prob]
    filtered_df = filtered_df.sort_values(by="Hit Probability (%)", ascending=False).reset_index(drop=True)
    
    # KPIs Layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Players Analyzed", len(df_predictions))
    with col2:
        st.metric("Highest Hit Chance Today", f"{filtered_df['Hit Probability (%)'].max()}%" if not filtered_df.empty else "N/A")
    with col3:
        st.metric("Players Above Threshold", len(filtered_df))
        
    st.markdown("---")
    
    # Search Box
    search_query = st.text_input("🔍 Search Hitter or Team:")
    if search_query:
        filtered_df = filtered_df[
            filtered_df['Player'].str.contains(search_query, case=False) | 
            filtered_df['Team'].str.contains(search_query, case=False)
        ]

    # Main Interactive Data Table
    st.subheader(f"Hit Projections for {date_formatted}")
    st.dataframe(
        filtered_df,
        column_config={
            "Hit Probability (%)": st.column_config.ProgressColumn(
                "Hit Probability (%)",
                help="The mathematical likelihood of getting >= 1 hit.",
                format="%f%%",
                min_value=0,
                max_value=100,
            ),
            "Season BA": st.column_config.NumberColumn("Season BA", format="%.3f")
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.warning("No data found or lineups haven't been posted yet for this date. Try shifting the date picker.")
