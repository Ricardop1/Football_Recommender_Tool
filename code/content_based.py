import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from paretoset import paretoset
import pycountry

from .cols_constant import *

def get_full_stats(minutes):
    full_data = pd.read_csv("./data/full_data_2022.csv")
    full_data = full_data[(full_data['Pos'] != 'GK') & (full_data["MP"] >= minutes)].reset_index()
    full_data.drop(["index","MP"],axis = 1, inplace = True)

    return full_data


def get_allstats(minutes):
    # cargamos todos los datos estadisticos relativos a las distintas caracteristicas de los jugadores
    def_df_full = pd.read_csv("./data/defensive_2022.csv")
    pass_df_full = pd.read_csv("./data/passing_2022.csv")
    poss_df_full = pd.read_csv("./data/possession_2022.csv")
    shoot_df_full = pd.read_csv("./data/shooting_2022.csv")
    goalshoot_df_full = pd.read_csv("./data/goal_shot_creation_2022.csv")
    passingType_df_full = pd.read_csv("./data/passing_type_2022.csv")

    # se eliminan algunas variables comunes o que no interesen
    def_df_full.drop(["Rk", "Born", "Matches", "Pressure_%", "vsDriblers_Tkl%", "Tkl+Int", "test"], axis=1,
                     inplace=True)

    pass_df_full.drop(
        ["Rk", "Player", "Squad", "Comp", "Born", "Matches", "Age", "Pos", "Nation", "90s", "Ttl_Pass_Cmpl%",
         "Short_Pass_Cmpl%", "Medium_Pass_Cmp%", "Long_Pass_Cmp%", "A-xA", "test"],
        axis=1, inplace=True)

    poss_df_full["Poss_Fail"] = poss_df_full["Miss_Poss"] + poss_df_full["Tackled"]
    poss_df_full.drop(["Rk", "Player", "Squad", "Comp", "Born", "Matches", "Age", "Pos", "Nation", "90s",
                       "Megs", "Dribbles_Succ%", "Tackled", "Miss_Poss", "Succ_RecPass%", "test"
                       ], axis=1, inplace=True)

    shoot_df_full.drop(["Rk", "Player", "Squad", "Comp", "Born", "Matches", "Age", "Pos", "Nation", "90s",
                        "SoT%", "Sh/90", "SoT/90", "FK", "PK", "PKatt", "npxG", "G-xG", "np:G-xG", "test"], axis=1,
                       inplace=True)

    goalshoot_df_full.drop(["Rk", "Player", "Squad", "Comp", "Born", "Matches", "Age", "Pos", "Nation", "90s",
                            "SCA90", "GCA90", "test"], axis=1, inplace=True)

    passingType_df_full.drop(["Rk", "Player", "Squad", "Comp", "Born", "Matches", "Age", "Pos", "Nation", "90s",
                              "Att", "Live", "Dead", "FK", "CK", "Cmp", "Off", "Out", "Int", "Blocks", "test"], axis=1,
                             inplace=True)

    # realizamo un preprocesado a algunas columnas de interés
    def_df_full['Pos'] = def_df_full['Pos'].str[0:2]
    def_df_full['Nation'] = def_df_full['Nation'].str[0:2]
    def_df_full["Player"] = def_df_full["Player"].str.split("\\", n=1, expand=True)[0]

    # juntamos todos los conjuntos de datos en uno mismo
    allstats_df = pd.concat(
        [def_df_full, pass_df_full, poss_df_full, shoot_df_full, goalshoot_df_full, passingType_df_full], axis=1)
    # filtramos por aquellos jugadores que no sean porteros
    allstats_df["MP"] = allstats_df["90s"] * 90
    allstats_df.fillna(0, inplace=True)
    allstats_df = allstats_df[(allstats_df['Pos'] != 'GK') & (allstats_df["MP"] >= minutes)].reset_index()

    allstats_df.drop(["index", "MP"], axis=1, inplace=True)

    return allstats_df


def plot_similar_players(player1, player2, df):
    categories = df.iloc[:, 1:].columns.values.tolist()
    player1_stats = df[df["Player"] == player1]
    player1_stats = player1_stats.iloc[:, 1:].copy()
    player2_stats = df[df["Player"] == player2]
    player2_stats = player2_stats.iloc[:, 1:].copy()

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=list(player1_stats.values.squeeze()),
        theta=categories,
        fill='toself',
        name=player1

    ))
    fig.add_trace(go.Scatterpolar(
        r=list(player2_stats.values.squeeze()),
        theta=categories,
        fill='toself',
        name=player2
    ))

    fig.update_layout(title={
        'text': player1 + " vs " + player2,
        'x': 0.5
    },
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig


def index_from_name(df, name):
    return df[df['Player'] == name].index.values[0]


def name_from_index(df, index):
    return df[df.index == index].Player.values[0]


def recommendations(name, df, cosine_matrix, number_of_recommendations):
    index = index_from_name(df, name)

    similarity_scores = list(enumerate(cosine_matrix[index]))
    similarity_scores_sorted = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations_indices = [t[0] for t in similarity_scores_sorted[1:(number_of_recommendations + 1)]]

    return df['Player'].iloc[recommendations_indices], similarity_scores_sorted[1:(number_of_recommendations + 1)]


def get_recommendations_by_player(name_player, df, number_of_recommendations):
    scaler = StandardScaler()
    df_Stats = df.iloc[:,8:]
    df_Stats_scaled = scaler.fit_transform(df_Stats)

    cosine_matrix = cosine_similarity(df_Stats_scaled)
    return recommendations(name_player, df, cosine_matrix, number_of_recommendations)


def get_positions_df(all_stats):
    # obtenemos las variables estadisticas para calcular la correlacion con la posicion

    cols_to_corr = all_stats.iloc[:, 8:].columns.values.tolist()
    cols_to_corr.append("Pos")

    # hacemos one hot encoding de la variable POS
    df_to_corr = all_stats[cols_to_corr]
    one_hot = pd.get_dummies(df_to_corr['Pos'])
    df_to_corr = df_to_corr.drop('Pos', axis=1)
    df_to_corr = df_to_corr.join(one_hot)

    # obtenemos la correlación entre todas las variables y obtenemos las columnas de cada posicion
    corr_pos = df_to_corr.corr().abs()
    corr_pos = corr_pos[["DF", "MF", "FW"]]

    # obtenemos las variables con una correlación en valor absoluto mayor que 0.5 con cada posición
    # en el caso de la posicion MF añadimos algunas variables debido a la complejidad de dcha posicion
    high_corr_df = corr_pos[corr_pos["DF"] > 0.5].index.values
    high_corr_df = high_corr_df[[x not in ["DF", "MF", "FW"] for x in high_corr_df]]

    high_corr_mf = corr_pos[corr_pos["MF"] > 0.4].index.values
    high_corr_mf = high_corr_mf[[x not in ["DF", "MF", "FW"] for x in high_corr_mf]]
    mf_feat = []
    high_corr_mf = np.unique(high_corr_mf.tolist() + mf_feat)

    high_corr_fw = corr_pos[corr_pos["FW"] > 0.5].index.values
    high_corr_fw = high_corr_fw[[x not in ["DF", "MF", "FW"] for x in high_corr_fw]]

    defensive_df = all_stats[list(high_corr_df) + ["Player", "Squad", "Pos", "Nation"]]
    defensive_df = defensive_df[defensive_df["Pos"] == "DF"]

    midfield_df = all_stats[list(high_corr_mf) + ["Player", "Squad", "Pos", "Nation"]]
    midfield_df = midfield_df[midfield_df["Pos"] == "MF"]

    forward_df = all_stats[list(high_corr_fw) + ["Player", "Squad", "Pos", "Nation"]]
    forward_df = forward_df[forward_df["Pos"] == "FW"]

    dict_pos = {
        "DEF": [defensive_df, high_corr_df],
        "MED": [midfield_df, high_corr_mf],
        "ATT": [forward_df, high_corr_fw]
    }

    return dict_pos



def get_full_positions_df(all_stats):

    high_corr_df = np.unique(DF_COLS + DF_CB_COLS + DF_DB_COLS)
    high_corr_mf = np.unique(MF_COLS + MF_CM_COLS + MF_DM_COLS + MF_AM_COLS)
    high_corr_fw = np.unique(FW_COLS + FW_AW_COLS + FW_ST_COLS)

    defensive_df = all_stats[list(high_corr_df) + ["Player", "Squad", "Pos", "Nation"]]
    defensive_df = defensive_df[defensive_df["Pos"] == "DF"]

    midfield_df = all_stats[list(high_corr_mf) + ["Player", "Squad", "Pos", "Nation"]]
    midfield_df = midfield_df[midfield_df["Pos"] == "MF"]

    forward_df = all_stats[list(high_corr_fw) + ["Player", "Squad", "Pos", "Nation"]]
    forward_df = forward_df[forward_df["Pos"] == "FW"]

    dict_pos = {
        "DEF": [defensive_df, high_corr_df],
        "MED": [midfield_df, high_corr_mf],
        "ATT": [forward_df, high_corr_fw]
    }

    return dict_pos

def get_recommendation_by_pos_team(team, pos, num, all_stats):
    df_pos, cols_pos = get_full_positions_df(all_stats)[pos]

    by_team_pos = df_pos[list(cols_pos) + ["Squad"]].groupby("Squad").mean().reset_index()
    by_team_pos["Player"] = by_team_pos["Squad"]
    team_row = by_team_pos[by_team_pos["Squad"] == team]

    df_pos = df_pos[df_pos["Squad"] != team]
    df_team = pd.concat([df_pos, team_row]).reset_index()
    df_team.drop(["index"], axis=1, inplace=True)

    df_stats = df_team[cols_pos]

    scaler = StandardScaler()
    df_Stats_scaled = scaler.fit_transform(df_stats)
    cosine_matrix = cosine_similarity(df_Stats_scaled)

    return recommendations(team, df_team, cosine_matrix, num)


def get_plot_df(all_stats):
    # escalamos las variables para la posterior visualziacion

    stats_toscale = all_stats.iloc[:, np.r_[0, 7:all_stats.shape[1]]]
    only_stats = stats_toscale.iloc[:, 1:]
    only_stats = (only_stats - only_stats.min()) / (only_stats.max() - only_stats.min())
    plot_players = pd.concat([stats_toscale.iloc[:, 0], only_stats], axis=1)
    return plot_players


def get_scaled_df_with_team(pos, team, all_stats):
    scaler = MinMaxScaler()

    df_pos, cols_pos = get_positions_df(all_stats)[pos]

    by_team_pos = df_pos[list(cols_pos) + ["Squad"]].groupby("Squad").mean().reset_index()
    by_team_pos["Player"] = by_team_pos["Squad"]
    team_row = by_team_pos[by_team_pos["Squad"] == team]

    df_team = pd.concat([df_pos, team_row]).reset_index()
    df_team.drop(["index"], axis=1, inplace=True)

    df_team[cols_pos] = scaler.fit_transform(df_team[cols_pos])

    return df_team[["Player"] + list(cols_pos)]


def get_national_team(nation, all_stats):
    all_stats["Nation"] = all_stats.apply(lambda x: pycountry.countries.get(alpha_2=x.Nation.upper()).name if pycountry.countries.get(alpha_2=x.Nation.upper()) else x.Nation, axis = 1)
    pos_df = get_positions_df(all_stats)

    data_def, col_def = pos_df["DEF"]
    data_def = data_def.loc[data_def.Nation == nation]

    data_mid, col_mid = pos_df["MED"]
    data_mid = data_mid.loc[data_mid.Nation == nation]

    data_att, col_att = pos_df["ATT"]
    data_att = data_att.loc[data_att.Nation == nation]

    st.write(all_stats[all_stats.Nation == nation].shape)

    def_weigts=["max", "max","max","max","max","max",
     "max","max","max","max","max","max",
     "min","max","max","max","max","max",
     "max","max","max"]
    def_weigts = ["max" for i in col_def]
    mid_weights = ["max", "max","max","max","max","max",
                   "max","max","max","max","max","max",
                   "max","max","max","max","max","max",]
    mid_weights = ["max" for i in col_mid]
    att_weights = ["max", "max","max","max","max","max",
                   "min","max","max","max","max","max",
                   "min","max","min","max","max","max",
                   "max","max","max","min","max","max",
                   "max","max","max"]
    att_weights = ["max" for i in col_att]
    mask_def = paretoset(data_def[col_def], sense=def_weigts)
    mask_mid = paretoset(data_mid[col_mid], sense=mid_weights)
    mask_att = paretoset(data_att[col_att], sense=att_weights)

    paretoset_def = data_def[mask_def]
    paretoset_mid = data_mid[mask_mid]
    paretoset_att = data_att[mask_att]

    def_crit = ["+" if w == "max" else "-" for w in def_weigts]
    mid_crit = ["+" if w == "max" else "-" for w in mid_weights]
    att_crit = ["+" if w == "max" else "-" for w in att_weights]

    def_weigts = [1 for w in def_weigts]
    mid_weights = [1 for w in mid_weights]
    att_weights = [1 for w in att_weights]

    paretoset_def["Rank"] = calc_topsis(data_def[col_def],len(col_def),def_weigts, def_crit)["Rank"]
    paretoset_def = paretoset_def.sort_values(by=['Rank'])

    paretoset_mid["Rank"] = calc_topsis(data_mid[col_mid],len(col_mid),mid_weights, mid_crit)["Rank"]
    paretoset_mid = paretoset_mid.sort_values(by=['Rank'])

    paretoset_att["Rank"] = calc_topsis(data_att[col_att],len(col_att),att_weights, att_crit)["Rank"]
    paretoset_att = paretoset_att.sort_values(by=['Rank'])

    return paretoset_def[["Player","Nation"]], paretoset_mid[["Player","Nation"]], paretoset_att[["Player","Nation"]]

def Normalize(dataset, nCol, weights):
    for i in range(1, nCol):
        temp = 0
        # Calculating Root of Sum of squares of a particular column
        for j in range(len(dataset)):
            temp = temp + dataset.iloc[j, i]**2
        temp = temp**0.5
        # Weighted Normalizing a element
        for j in range(len(dataset)):
            dataset.iat[j, i] = (dataset.iloc[j, i] / temp)*weights[i-1]
    return dataset

# Calculate ideal best and ideal worst
def Calc_Values(dataset, nCol, impact):
    p_sln = (dataset.max().values)[1:]
    n_sln = (dataset.min().values)[1:]
    for i in range(1, nCol):
        if impact[i-1] == '-':
            p_sln[i-1], n_sln[i-1] = n_sln[i-1], p_sln[i-1]
    return p_sln, n_sln

def calc_topsis(dataset, ncol, weights, impact):
    temp_dataset = Normalize(dataset, ncol, weights)
    p_sln, n_sln = Calc_Values(temp_dataset, ncol, impact)

    # calculating topsis score
    score = [] # Topsis score
    pp = [] # distance positive
    nn = [] # distance negative


    # Calculating distances and Topsis score for each row
    for i in range(len(temp_dataset)):
        temp_p, temp_n = 0, 0
        for j in range(1, ncol):
            temp_p = temp_p + (p_sln[j-1] - temp_dataset.iloc[i, j])**2
            temp_n = temp_n + (n_sln[j-1] - temp_dataset.iloc[i, j])**2
        temp_p, temp_n = temp_p**0.5, temp_n**0.5
        score.append(temp_n/(temp_p + temp_n))
        nn.append(temp_n)
        pp.append(temp_p)

    dataset['distance positive'] = pp
    dataset['distance negative'] = nn
    dataset['Topsis Score'] = score

    # calculating the rank according to topsis score
    dataset['Rank'] = (dataset['Topsis Score'].rank(method='max', ascending=False))
    dataset = dataset.astype({"Rank": int})
    return dataset