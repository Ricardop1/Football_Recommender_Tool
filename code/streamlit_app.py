import numpy as np
import streamlit as st
from content_based import *
from collaborative_filtering import *
import pandas as pd

st.title('Football recommender Tool')

all_stats = get_allstats(500)
players_basic_info = all_stats.iloc[:, :7]

model = get_model()
joint_df = load_joint_df()

stats_to_scale = all_stats.iloc[:, np.r_[0, 7:all_stats.shape[1]]]
only_stats = stats_to_scale.iloc[:, 1:]
only_stats = (only_stats - only_stats.min()) / (only_stats.max() - only_stats.min())
plot_players = pd.concat([stats_to_scale.iloc[:, 0], only_stats], axis=1)

unique_players = players_basic_info["Player"].sort_values(ascending=True).unique()
unique_players = np.insert(unique_players, 0, "Select Option")
unique_teams = players_basic_info["Squad"].sort_values(ascending=True).unique()
unique_teams = np.insert(unique_teams, 0, "Select Option")


col1, col2 = st.columns([1, 1])
with col1:
    select_type = st.selectbox("Select Recommender Type", ["Similar to Player",
                                                           "Similar to Team",
                                                           "Best players to fit a Team"], key="select_reco")
with col2:
    if select_type == "Similar to Player":
        st.selectbox("Select Player", unique_players,  key="select_player")
    else:
        st.selectbox("Select Team", unique_teams, key="select_team")


if "select_player" in st.session_state and st.session_state.select_player != "Select Option":
    st.write(f"Most similar players to {st.session_state.select_player}:")

    jugadores, score = get_recommendations_by_player(st.session_state.select_player, all_stats, 10)
    df_res = pd.DataFrame({"Player": jugadores, "Score": [i[1] for i in score]}).reset_index()
    st.table(df_res[["Player", "Score"]])

    player_plot = st.selectbox("Select Player to Visualize", jugadores, key="select_player_visualize1")

    fig_play = plot_similar_players(player_plot, st.session_state.select_player, plot_players)
    st.plotly_chart(fig_play, use_container_width=True)

elif "select_team" in st.session_state and \
        select_type == "Similar to Team" and \
        st.session_state.select_team != "Select Option":

    position = st.selectbox("Select Position", ["DEF", "MED", "ATT"])

    st.write(f"Most similar players to {st.session_state.select_team} on {position}:")
    jugadores, score = get_recommendation_by_pos_team(st.session_state.select_team, position, 10, all_stats)
    df_res_team = pd.DataFrame({"Player": jugadores, "Score": [i[1] for i in score]}).reset_index()
    st.table(df_res_team[["Player", "Score"]])

    df_with_team = get_scaled_df_with_team(position, st.session_state.select_team, all_stats)
    player_plot = st.selectbox("Select Player to Visualize", jugadores, key="select_player_visualize2")
    fig_team = plot_similar_players(player_plot, st.session_state.select_team, df_with_team)
    st.plotly_chart(fig_team, use_container_width=True)
elif "select_team" in st.session_state and \
        select_type == "Best players to fit a Team" and \
        st.session_state.select_team != "Select Option":

    current_df, recommender_df = get_recommendation_model(st.session_state.select_team,model, joint_df)
    st.write(f"Best 5 current players that played in {st.session_state.select_team}:")
    st.table(current_df)
    st.write(f"Best fit players to {st.session_state.select_team}:")
    st.table(recommender_df)

