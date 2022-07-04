from content_based import *
import pandas as pd

all_stats = get_allstats(800)

# obtenemos las recomendaciones de jugadores similares para un jugador concreto
player_to_compare = "Karim Benzema"

jugadores, score = get_recommendations_by_player(player_to_compare, all_stats, 10)

df_res = pd.DataFrame({"Player": jugadores, "Score": [i[1] for i in score]}).reset_index()



# obtener recomendaciones para un equipo dado y una posicion
team = "Real Madrid"

jugadores, score = get_recommendation_by_pos_team(team, "ATT", 10, all_stats)

df_res_team = pd.DataFrame({"Player": jugadores, "Score": [i[1] for i in score]}).reset_index()

df_with_team = get_scaled_df_with_team("ATT", team, all_stats)
plot_similar_players("Musa Barrow", team, df_with_team)
