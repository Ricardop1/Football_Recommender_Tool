import pandas as pd
import numpy as np
import streamlit as st
from ast import literal_eval
from content_based import get_allstats
from tensorflow import keras


def clean_clubs(clubs):
    if isinstance(clubs, str):
        data = clubs.replace("\n", "")
        data = data.replace("\' \'", "\',\'")
        data = data.replace("\' \"", "\',\"")
        data = literal_eval(data)
    else:
        data = ''.join(clubs)
        data = clubs.replace("\n", "")
        data = data.replace("\' \'", "\',\'")
        data = data.replace("\' \"", "\',\"")

    return data


def read_all_data_clubs_ratings():
    players_clubs = pd.read_csv("./data/final_df.csv")
    # aplicamos la función cleanClubs
    players_clubs["clubs_list"] = players_clubs.apply(lambda x: clean_clubs(x.all_clubs), axis=1)
    # creamos una fila para jugador y equipo en el que ha jugado
    df_all_players_teams = players_clubs.explode("clubs_list").reset_index()
    # eliminamos algunas variables
    df_all_players_teams.drop(["all_clubs", "index"], axis=1, inplace=True)
    # eliminamos aquellos equipo que solamente aparezcan 2 veces
    counts = df_all_players_teams['clubs_list'].value_counts()
    result_df = df_all_players_teams[~df_all_players_teams['clubs_list'].isin(counts[counts < 2].index)].copy()
    result_df["Rating"] = 1

    all_stats_df = get_allstats(0)
    unique_teams = all_stats_df.Squad.unique()
    rating_players = result_df[result_df['clubs_list'].isin(unique_teams)]

    # obtenemos las columnas que representen datos estadisticos
    column_stats = all_stats_df.iloc[:, 7:].columns
    columnas_rating = []

    for col in column_stats:
        new_col = col + "_rating"
        columnas_rating.append(new_col)
        p25 = np.percentile(all_stats_df[col], 25)
        p50 = np.percentile(all_stats_df[col], 50)
        p75 = np.percentile(all_stats_df[col], 75)
        p90 = np.percentile(all_stats_df[col], 90)
        all_stats_df[new_col] = all_stats_df[col].apply(lambda x: getRating(x, [p25, p50, p75, p90], col))

    # obtenemos una columa con la lista de las valoraciones
    all_stats_df['ratings_combined'] = all_stats_df[columnas_rating].values.tolist()
    all_stats_df.drop(columnas_rating, axis=1, inplace=True)

    player_club_rating = rating_players.merge(all_stats_df[["Player", "ratings_combined"]], on='Player', how='left')
    player_club_rating = player_club_rating.dropna()

    # aplicamos el mismo proceso pero a cada equipo agrupando por Squad
    df_Stats = all_stats_df.iloc[:, 7:]
    cols = df_Stats.columns.values.tolist()
    cols.append("Squad")

    byTeam = all_stats_df[cols].groupby("Squad").mean().reset_index()
    columnas_rating = []

    for col in column_stats:
        new_col = col + "_rating"
        columnas_rating.append(new_col)
        p25 = np.percentile(byTeam[col], 25)
        p50 = np.percentile(byTeam[col], 50)
        p75 = np.percentile(byTeam[col], 75)
        p90 = np.percentile(byTeam[col], 90)
        byTeam[new_col] = byTeam[col].apply(lambda x: getRating(x, [p25, p50, p75, p90], col))

    byTeam['ratingsTeam_combined'] = byTeam[columnas_rating].values.tolist()
    byTeam.drop(columnas_rating, axis=1, inplace=True)

    player_club_rating.rename(columns={"clubs_list": "Squad"}, inplace=True)
    player_club_rating = player_club_rating.merge(byTeam[["Squad", "ratingsTeam_combined"]], on='Squad', how='left')

    # aplicamos una media ponderada de forma a obtener una valoracion de cada equipo a cada ugador que alguna vez jugó en dicho equipo
    player_club_rating["Rating"] = player_club_rating.apply(
        lambda x: np.average(x.ratings_combined, weights=x.ratingsTeam_combined), axis=1)
    player_club_rating = player_club_rating[["Player", "Nation", "Squad", "Rating"]]

    return player_club_rating


def invertRating(rat, min_rat, max_rat):
    """
    funcion para invertir una valoración
    """
    return (max_rat - rat) + min_rat


def getRating(feat_value, percentils, col):
    """
    funcion para obtener la valoracion de un jugador en base a los percentiles de una columna
    :param feat_value:
    :param percentils:
    :param col:
    :return:
    """
    rating = 0
    inverseRats = ["Poss_Fail", "Err"]
    if feat_value > percentils[3]:
        rating = 5
    elif feat_value > percentils[2]:
        rating = 4
    elif feat_value > percentils[1]:
        rating = 3
    elif feat_value > percentils[0]:
        rating = 2
    else:
        rating = 1
    if col in inverseRats:
        rating = invertRating(rating, 0, 5)
    return rating


@st.cache
def get_model():
    return keras.models.load_model('./recommender_bp_model')


@st.cache
def load_joint_df():
    return pd.read_csv("./data/joint_df.csv")


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
