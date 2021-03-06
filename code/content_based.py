import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go


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

    # realizamo un preprocesado a algunas columnas de inter??s
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
    df_stats = df.iloc[:, 7:]
    cosine_matrix = cosine_similarity(df_stats, df_stats)
    return recommendations(name_player, df, cosine_matrix, number_of_recommendations)


def get_positions_df(all_stats):
    # obtenemos las variables estadisticas para calcular la correlacion con la posicion

    cols_to_corr = all_stats.iloc[:, 7:].columns.values.tolist()
    cols_to_corr.append("Pos")

    # hacemos one hot encoding de la variable POS
    df_to_corr = all_stats[cols_to_corr]
    one_hot = pd.get_dummies(df_to_corr['Pos'])
    df_to_corr = df_to_corr.drop('Pos', axis=1)
    df_to_corr = df_to_corr.join(one_hot)

    # obtenemos la correlaci??n entre todas las variables y obtenemos las columnas de cada posicion
    corr_pos = df_to_corr.corr().abs()
    corr_pos = corr_pos[["DF", "MF", "FW"]]

    # obtenemos las variables con una correlaci??n en valor absoluto mayor que 0.5 con cada posici??n
    # en el caso de la posicion MF a??adimos algunas variables debido a la complejidad de dcha posicion
    high_corr_df = corr_pos[corr_pos["DF"] > 0.5].index.values
    high_corr_df = high_corr_df[[x not in ["DF", "MF", "FW"] for x in high_corr_df]]

    high_corr_mf = corr_pos[corr_pos["MF"] > 0.4].index.values
    high_corr_mf = high_corr_mf[[x not in ["DF", "MF", "FW"] for x in high_corr_mf]]
    mf_feat = ["Pressure_Press", "Pressure_Def 3rd", "Pressure_Mid 3rd", "Pressure_Att 3rd",
               "tackles_Def 3rd", "tackles_Mid 3rd", "tackles_Att 3rd", "Blocks", "SCA", "GCA",
               "Ttl_Pass_Compl", "Ttl_Dist_Pass", "Prgsv_Dist", "Touches_Def3rd", "Touches_Mid3rd",
               "Touches_Att3rd", "Touches_Def3rd", "Touches_Mid3rd", "Touches_Att3rd"]
    high_corr_mf = np.unique(high_corr_mf.tolist() + mf_feat)

    high_corr_fw = corr_pos[corr_pos["FW"] > 0.5].index.values
    high_corr_fw = high_corr_fw[[x not in ["DF", "MF", "FW"] for x in high_corr_fw]]

    defensive_df = all_stats[list(high_corr_df) + ["Player", "Squad", "Pos"]]
    defensive_df = defensive_df[defensive_df["Pos"] == "DF"]

    midfield_df = all_stats[list(high_corr_mf) + ["Player", "Squad", "Pos"]]
    midfield_df = midfield_df[midfield_df["Pos"] == "MF"]

    forward_df = all_stats[list(high_corr_fw) + ["Player", "Squad", "Pos"]]
    forward_df = forward_df[forward_df["Pos"] == "FW"]

    dict_pos = {
        "DEF": [defensive_df, high_corr_df],
        "MED": [midfield_df, high_corr_mf],
        "ATT": [forward_df, high_corr_fw]
    }

    return dict_pos


def get_recommendation_by_pos_team(team, pos, num, all_stats):
    df_pos, cols_pos = get_positions_df(all_stats)[pos]

    by_team_pos = df_pos[list(cols_pos) + ["Squad"]].groupby("Squad").mean().reset_index()
    by_team_pos["Player"] = by_team_pos["Squad"]
    team_row = by_team_pos[by_team_pos["Squad"] == team]

    df_pos = df_pos[df_pos["Squad"] != team]
    df_team = pd.concat([df_pos, team_row]).reset_index()
    df_team.drop(["index"], axis=1, inplace=True)

    df_stats = df_team[cols_pos]
    cosine_matrix = cosine_similarity(df_stats, df_stats)

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
