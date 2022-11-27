import pandas as pd
import numpy as np
import streamlit as st
from tensorflow import keras


@st.cache(allow_output_mutation=True)
def get_model():
    return keras.models.load_model('./data/recommender_bp_model_2022')


@st.cache
def load_joint_df():
    return pd.read_csv("./data/nation_joint_df_2022.csv")


def get_recommendation_model(team, model, df):
    user_ids = df["Squad"].unique().tolist()
    player_ids = df["Player"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    player2player_encoded = {x: i for i, x in enumerate(player_ids)}
    player_encoded2player = {i: x for i, x in enumerate(player_ids)}

    user_id = team
    players_watched_by_user = df[df.Squad == user_id]
    players_not_watched = df[
        ~df["Player"].isin(players_watched_by_user.Player.values)
    ]["Player"]
    players_not_watched = list(
        set(players_not_watched).intersection(set(player2player_encoded.keys()))
    )
    players_not_watched = [[player2player_encoded.get(x)] for x in players_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_player_array = np.hstack(
        ([[user_encoder]] * len(players_not_watched), players_not_watched)
    )
    ratings = model.predict(user_player_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_player_ids = [
        player_encoded2player.get(players_not_watched[x][0]) for x in top_ratings_indices
    ]

    top_players_user = (
        players_watched_by_user.sort_values(by="rating", ascending=False)
            .head(5)
            .Player.values
    )
    player_df_rows = df[df["Player"].isin(top_players_user)]
    player_df_rows = player_df_rows.drop_duplicates(subset=["player"])
    player_df_rows = player_df_rows.sort_values(by=['rating'], ascending=False).reset_index()
    player_df_rows.drop(["index"], axis=1, inplace=True)
    player_df_rows = player_df_rows[["Player", "Nation", "Rating"]]

    recommended_players = df[df["Player"].isin(recommended_player_ids)]
    recommended_players = recommended_players.drop_duplicates(subset=["player"])
    recommended_players = recommended_players.sort_values(by=['rating'], ascending=False)
    recommended_players = recommended_players[["Player", "Nation", "Rating"]].reset_index()
    recommended_players.drop(["index"], axis=1, inplace=True)

    return player_df_rows, recommended_players
