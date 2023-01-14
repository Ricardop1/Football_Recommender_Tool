import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from paretoset import paretoset
import pycountry
from mplsoccer import PyPizza, FontManager

from cols_constant import *

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


def plot_similar_players_pizza(player1, player2, df):
    params = df.iloc[:, 1:].columns.values.tolist()
    player1_stats = df[df["Player"] == player1]
    player1_stats = player1_stats.iloc[:, 1:].copy().values.squeeze()
    player2_stats = df[df["Player"] == player2]
    player2_stats = player2_stats.iloc[:, 1:].copy().values.squeeze()
    #min_range = [i / 2 for i in player1_stats]
    min_range = player1_stats / 2
    max_range = player1_stats * 2
    #max_range = [i * 2 for i in player1_stats]
    #st.write(min_range.tolist())
    font_normal = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/roboto/'
                              'Roboto%5Bwdth,wght%5D.ttf')


    baker = PyPizza(
        params=params,
        min_range = min_range.tolist()[0],
        max_range = max_range.tolist()[0],
        background_color="#222222", straight_line_color="#000000",
        last_circle_color="#000000", last_circle_lw=2.5, other_circle_lw=0,
        other_circle_color="#000000", straight_line_lw=1
    )

    # plot pizza
    fig, ax = baker.make_pizza(
        player1_stats,                     # list of values
        compare_values=player2_stats,    # passing comparison values
        figsize=(8, 8),             # adjust figsize according to your need
        color_blank_space="same",   # use same color to fill blank space
        blank_alpha=0.4,            # alpha for blank-space colors
        param_location=110,         # where the parameters will be added
        kwargs_slices=dict(
            facecolor="#1A78CF", edgecolor="#000000",
            zorder=1, linewidth=1
        ),                          # values to be used when plotting slices
        kwargs_compare=dict(
            facecolor="#ff9300", edgecolor="#222222", zorder=3, linewidth=1,
        ),                          # values to be used when plotting comparison slices
        kwargs_params=dict(
            color="#F2F2F2", fontsize=12, zorder=5,
            fontproperties=font_normal.prop, va="center"
        ),                          # values to be used when adding parameter
        kwargs_values=dict(
            color="#000000", fontsize=12,
            fontproperties=font_normal.prop, zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="#1A78CF",
                boxstyle="round,pad=0.2", lw=1
            )
        ),                           # values to be used when adding parameter-values
        kwargs_compare_values=dict(
            color="#000000", fontsize=12,
            fontproperties=font_normal.prop, zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="#FF9300",
                boxstyle="round,pad=0.2", lw=1
            )
        )                            # values to be used when adding comparison-values
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


def get_positions_df_national_team(all_stats):
    high_corr_df_cb = np.unique(DF_CB_COLS)
    high_corr_df_db = np.unique(DF_DB_COLS)

    high_corr_mf_dm = np.unique(MF_DM_COLS)
    high_corr_mf_cm = np.unique(MF_CM_COLS )
    high_corr_mf_am = np.unique(MF_AM_COLS)

    high_corr_fw_aw = np.unique(FW_AW_COLS)
    high_corr_fw_st = np.unique(FW_ST_COLS)

    defensive_df_cb = all_stats[list(high_corr_df_cb) + ["Player", "Squad", "Pos", "Nation"]]
    defensive_df_cb = defensive_df_cb[defensive_df_cb["Pos"] == "DF"]
    defensive_df_db = all_stats[list(high_corr_df_db) + ["Player", "Squad", "Pos", "Nation"]]
    defensive_df_db = defensive_df_db[defensive_df_db["Pos"] == "DF"]

    defensive_mf_dm = all_stats[list(high_corr_mf_dm) + ["Player", "Squad", "Pos", "Nation"]]
    defensive_mf_dm = defensive_mf_dm[defensive_mf_dm["Pos"] == "MF"]
    defensive_mf_cm = all_stats[list(high_corr_mf_cm) + ["Player", "Squad", "Pos", "Nation"]]
    defensive_mf_cm = defensive_mf_cm[defensive_mf_cm["Pos"] == "MF"]
    defensive_mf_am = all_stats[list(high_corr_mf_am) + ["Player", "Squad", "Pos", "Nation"]]
    defensive_mf_am = defensive_mf_am[defensive_mf_am["Pos"] == "MF"]

    defensive_fw_aw = all_stats[list(high_corr_fw_aw) + ["Player", "Squad", "Pos", "Nation"]]
    defensive_fw_aw = defensive_fw_aw[defensive_fw_aw["Pos"] == "FW"]
    defensive_fw_st = all_stats[list(high_corr_fw_st) + ["Player", "Squad", "Pos", "Nation"]]
    defensive_fw_st = defensive_fw_st[defensive_fw_st["Pos"] == "FW"]

    dict_pos = {
        "CB": [defensive_df_cb, high_corr_df_cb],
        "DB": [defensive_df_db, high_corr_df_db],
        "CM": [defensive_mf_cm, high_corr_mf_cm],
        "DM": [defensive_mf_dm, high_corr_mf_dm],
        "AM": [defensive_mf_am, high_corr_mf_am],
        "AW": [defensive_fw_aw, high_corr_fw_aw],
        "ST": [defensive_fw_st, high_corr_fw_st]
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

    df_pos, cols_pos = get_full_positions_df(all_stats)[pos]

    by_team_pos = df_pos[list(cols_pos) + ["Squad"]].groupby("Squad").mean().reset_index()
    by_team_pos["Player"] = by_team_pos["Squad"]
    team_row = by_team_pos[by_team_pos["Squad"] == team]

    df_team = pd.concat([df_pos, team_row]).reset_index()
    df_team.drop(["index"], axis=1, inplace=True)

    df_team[cols_pos] = scaler.fit_transform(df_team[cols_pos])

    return df_team[["Player"] + list(cols_pos)]


def get_national_team(nation, all_stats):
    all_stats["Nation"] = all_stats.apply(lambda x: pycountry.countries.get(alpha_2=x.Nation.upper()).name if pycountry.countries.get(alpha_2=x.Nation.upper()) else x.Nation, axis = 1)
    pos_df = get_positions_df_national_team(all_stats)

    data_cb, col_cb = pos_df["CB"]
    data_cb = data_cb.loc[data_cb.Nation == nation]
    data_db, col_db = pos_df["DB"]
    data_db = data_db.loc[data_db.Nation == nation]

    data_dm, col_dm = pos_df["DM"]
    data_dm = data_dm.loc[data_dm.Nation == nation]
    data_cm, col_cm = pos_df["CM"]
    data_cm = data_cm.loc[data_cm.Nation == nation]
    data_am, col_am = pos_df["AM"]
    data_am = data_am.loc[data_am.Nation == nation]

    data_aw, col_aw = pos_df["AW"]
    data_aw = data_aw.loc[data_aw.Nation == nation]
    data_st, col_st = pos_df["ST"]
    data_st = data_st.loc[data_st.Nation == nation]

    def_weights=["max", "max","max","max","max","max",
     "max","max","max","max","max","max",
     "min","max","max","max","max","max",
     "max","max","max"]
    cb_weights = ["max" for i in col_cb]
    db_weights = ["max" for i in col_db]
    mid_weights = ["max", "max","max","max","max","max",
                   "max","max","max","max","max","max",
                   "max","max","max","max","max","max",]
    dm_weights = ["max" for i in col_dm]
    cm_weights = ["max" for i in col_cm]
    am_weights = ["max" for i in col_am]
    att_weights = ["max", "max","max","max","max","max",
                   "min","max","max","max","max","max",
                   "min","max","min","max","max","max",
                   "max","max","max","min","max","max",
                   "max","max","max"]
    aw_weights = ["max" for i in col_aw]
    st_weights = ["max" for i in col_st]

    mask_cb = paretoset(data_cb[col_cb], sense=cb_weights)
    mask_db = paretoset(data_db[col_db], sense=db_weights)
    mask_dm = paretoset(data_dm[col_dm], sense=dm_weights)
    mask_cm = paretoset(data_cm[col_cm], sense=cm_weights)
    mask_am = paretoset(data_am[col_am], sense=am_weights)
    mask_aw = paretoset(data_aw[col_aw], sense=aw_weights)
    mask_st = paretoset(data_st[col_st], sense=st_weights)

    paretoset_cb = data_cb[mask_cb]
    paretoset_db = data_db[mask_db]
    paretoset_dm = data_dm[mask_dm]
    paretoset_cm = data_cm[mask_cm]
    paretoset_am = data_am[mask_am]
    paretoset_aw = data_aw[mask_aw]
    paretoset_st = data_st[mask_st]

    cb_crit = ["+" if w == "max" else "-" for w in cb_weights]
    db_crit = ["+" if w == "max" else "-" for w in db_weights]
    dm_crit = ["+" if w == "max" else "-" for w in dm_weights]
    cm_crit = ["+" if w == "max" else "-" for w in cm_weights]
    am_crit = ["+" if w == "max" else "-" for w in am_weights]
    aw_crit = ["+" if w == "max" else "-" for w in aw_weights]
    st_crit = ["+" if w == "max" else "-" for w in st_weights]

    cb_weights = [1 for w in cb_weights]
    db_weights = [1 for w in db_weights]
    dm_weights = [1 for w in dm_weights]
    cm_weights = [1 for w in cm_weights]
    am_weights = [1 for w in am_weights]
    aw_weights = [1 for w in aw_weights]
    st_weights = [1 for w in st_weights]

    paretoset_cb["Rank"] = calc_topsis(data_cb[col_cb],len(col_cb),cb_weights, cb_crit)["Rank"]
    paretoset_cb = paretoset_cb.sort_values(by=['Rank'])
    paretoset_db["Rank"] = calc_topsis(data_db[col_db],len(col_db),db_weights, db_crit)["Rank"]
    paretoset_db = paretoset_db.sort_values(by=['Rank'])

    paretoset_dm["Rank"] = calc_topsis(data_dm[col_dm],len(col_dm),dm_weights, dm_crit)["Rank"]
    paretoset_dm = paretoset_dm.sort_values(by=['Rank'])
    paretoset_cm["Rank"] = calc_topsis(data_cm[col_cm],len(col_cm),cm_weights, cm_crit)["Rank"]
    paretoset_cm = paretoset_cm.sort_values(by=['Rank'])
    paretoset_am["Rank"] = calc_topsis(data_am[col_am],len(col_am),am_weights, am_crit)["Rank"]
    paretoset_am = paretoset_am.sort_values(by=['Rank'])

    paretoset_aw["Rank"] = calc_topsis(data_aw[col_aw],len(col_aw),aw_weights, aw_crit)["Rank"]
    paretoset_aw = paretoset_aw.sort_values(by=['Rank'])
    paretoset_st["Rank"] = calc_topsis(data_st[col_st],len(col_st),st_weights, st_crit)["Rank"]
    paretoset_st = paretoset_st.sort_values(by=['Rank'])

    return paretoset_cb[["Player","Nation"]], paretoset_db[["Player","Nation"]], paretoset_dm[["Player","Nation"]], \
           paretoset_cm[["Player","Nation"]], paretoset_am[["Player","Nation"]], paretoset_aw[["Player","Nation"]],\
           paretoset_st[["Player","Nation"]]

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