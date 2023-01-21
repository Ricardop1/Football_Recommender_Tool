import numpy as np
import streamlit as st
from content_based import *
from collaborative_filtering import *
import pandas as pd
import pycountry
from cols_constant import *


st.title('Football recommender Tool')
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

all_stats = get_full_stats(1000)
players_basic_info = all_stats.iloc[:, :8]

model = get_model()
joint_df = load_joint_df()

stats_to_scale = all_stats.iloc[:, np.r_[0, 8:all_stats.shape[1]]]
only_stats = stats_to_scale.iloc[:, 1:]
only_stats = (only_stats - only_stats.min()) / (only_stats.max() - only_stats.min())

plot_players = pd.concat([stats_to_scale.iloc[:, 0], only_stats], axis=1)

unique_players = players_basic_info["Player"].sort_values(ascending=True).unique()
unique_players = np.insert(unique_players, 0, "Select Option")
unique_teams = players_basic_info["Squad"].sort_values(ascending=True).unique()
unique_teams = np.insert(unique_teams, 0, "Select Option")
players_basic_info["Country"] = players_basic_info.apply(lambda x: pycountry.countries.get(alpha_2=x.Nation.upper()).name if pycountry.countries.get(alpha_2=x.Nation.upper()) else x.Nation, axis = 1)
unique_nations = players_basic_info["Country"].sort_values(ascending=True).unique()
unique_nations = np.insert(unique_nations, 0, "Select Option")


col1, col2 = st.columns([1, 1])
with col1:
    select_type = st.selectbox("Select Recommender Type", ["Similar to Player",
                                                           "Similar to Team",
                                                           "Best players to fit a Team",
                                                           "National Team Recommender"], key="select_reco")
with col2:
    if select_type == "Similar to Player":
        st.selectbox("Select Player", unique_players,  key="select_player")
    elif select_type == "National Team Recommender":
        st.selectbox("Select National Team: ", unique_nations,  key="select_nation")
    else:
        st.selectbox("Select Team", unique_teams, key="select_team")


if "select_player" in st.session_state and st.session_state.select_player != "Select Option":
    st.write(f"Most similar players to {st.session_state.select_player}:")

    jugadores, score = get_recommendations_by_player(st.session_state.select_player, all_stats, 10)
    df_res = pd.DataFrame({"Player": jugadores, "Score": [i[1] for i in score]}).reset_index()
    st.table(df_res[["Player", "Score"]])

    player_plot = st.selectbox("Select Player to Visualize", jugadores, key="select_player_visualize1")

    fig_play = plot_similar_players(player_plot, st.session_state.select_player, plot_players)
    mean_df = create_mean_df(player_plot)
    fig_play2 = plot_similar_players_pizza(player_plot, st.session_state.select_player, mean_df)

    st.plotly_chart(fig_play, use_container_width=True)
    st.write(fig_play2)


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
    fig_play2 = plot_similar_players_pizza(player_plot, st.session_state.select_team, df_with_team)
    st.plotly_chart(fig_team, use_container_width=True)
    st.write(fig_play2)
elif "select_team" in st.session_state and \
        select_type == "Best players to fit a Team" and \
        st.session_state.select_team != "Select Option":

    current_df, recommender_df = get_recommendation_model(st.session_state.select_team,model, joint_df)
    st.write(f"Best 5 current players that played in {st.session_state.select_team}:")
    st.table(current_df)
    st.write(f"Best fit players to {st.session_state.select_team}:")
    st.table(recommender_df)

elif "select_nation" in st.session_state and \
        select_type == "National Team Recommender" and \
        st.session_state.select_nation != "Select Option":

    cb_df, db_df, dm_df, cm_df, am_df, aw_df, st_df = get_national_team(st.session_state.select_nation, all_stats)
    full_list=[]
    st.write(f"Recommender Center Backs players for {st.session_state.select_nation}:")
    cb_df = cb_df[~cb_df.Player.isin(full_list)]
    st.table(cb_df.head(5))
    full_list.extend(cb_df.head(5).Player.values)

    st.write(f"Recommender Defensive Backs players for {st.session_state.select_nation}:")
    db_df = db_df[~db_df.Player.isin(full_list)]
    st.table(db_df.head(5))
    full_list.extend(db_df.head(5).Player.values)

    st.write(f"Recommender Defensive midfield players for {st.session_state.select_nation}:")
    dm_df = dm_df[~dm_df.Player.isin(full_list)]
    st.table(dm_df.head(5))
    full_list.extend(dm_df.head(5).Player.values)

    st.write(f"Recommender Center midfield players for {st.session_state.select_nation}:")
    cm_df = cm_df[~cm_df.Player.isin(full_list)]
    st.table(cm_df.head(5))
    full_list.extend(cm_df.head(5).Player.values)

    st.write(f"Recommender Attacking midfield players for {st.session_state.select_nation}:")
    am_df = am_df[~am_df.Player.isin(full_list)]
    st.table(am_df.head(5))
    full_list.extend(am_df.head(5).Player.values)

    st.write(f"Recommender Attack Winger players for {st.session_state.select_nation}:")
    aw_df = aw_df[~aw_df.Player.isin(full_list)]
    st.table(aw_df.head(5))
    full_list.extend(aw_df.head(5).Player.values)

    st.write(f"Recommender Stricker players for {st.session_state.select_nation}:")
    st_df = st_df[~st_df.Player.isin(full_list)]
    st.table(st_df.head(5))


