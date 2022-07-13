import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow import keras
from tensorflow.keras import layers
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from ast import literal_eval
from .content_based import get_allstats


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


@st.cache
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
