import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow import keras
from tensorflow.keras import layers

EMBEDDING_SIZE = 50


# creacion de la clase para el modelo de recomendación

@st.cache
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_players, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.player_encoded2player = None
        self.user2user_encoded = None
        self.player2player_encoded = None
        self.userencoded2user = None
        self.num_users = num_users
        self.num_players = num_players
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.player_embedding = layers.Embedding(
            num_players,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.player_bias = layers.Embedding(num_players, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        player_vector = self.player_embedding(inputs[:, 1])
        player_bias = self.player_bias(inputs[:, 1])
        dot_user_player = tf.tensordot(user_vector, player_vector, 2)
        # Add all the components (including bias)
        x = dot_user_player + user_bias + player_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)

    def train_model(self, player_club_ratings):
        user_ids = player_club_ratings["Squad"].unique().tolist()
        self.user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        self.userencoded2user = {i: x for i, x in enumerate(user_ids)}
        player_ids = player_club_ratings["Player"].unique().tolist()
        self.player2player_encoded = {x: i for i, x in enumerate(player_ids)}
        self.player_encoded2player = {i: x for i, x in enumerate(player_ids)}
        player_club_ratings["user"] = player_club_ratings["Squad"].map(self.user2user_encoded)
        player_club_ratings["player"] = player_club_ratings["Player"].map(self.player2player_encoded)

        num_users = len(self.user2user_encoded)
        num_players = len(self.player_encoded2player)
        player_club_ratings["rating"] = player_club_ratings["Rating"].values.astype(np.float32)
        # min and max ratings will be used to normalize the ratings later
        min_rating = min(player_club_ratings["rating"])
        max_rating = max(player_club_ratings["rating"])

        x = player_club_ratings[["user", "player"]].values
        # Normalize the targets between 0 and 1. Makes it easy to train.
        y = player_club_ratings["rating"].apply(lambda x: (x - 0) / (max_rating - 0)).values

        self.compile(
            loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.005)
        )
        history = self.fit(
            x=x,
            y=y,
            batch_size=64,
            epochs=5,
            verbose=0,
        )

        return self

    def get_recommendations(self, player_club_ratings, team):
        user_id = team
        players_watched_by_user = player_club_ratings[player_club_ratings.Squad == user_id]
        players_not_watched = player_club_ratings[
            ~player_club_ratings["Player"].isin(players_watched_by_user.Player.values)
        ]["Player"]
        players_not_watched = list(
            set(players_not_watched).intersection(set(self.player2player_encoded.keys()))
        )
        players_not_watched = [[self.player2player_encoded.get(x)] for x in players_not_watched]
        user_encoder = self.user2user_encoded.get(user_id)
        user_player_array = np.hstack(
            ([[user_encoder]] * len(players_not_watched), players_not_watched)
        )
        ratings = self.predict(user_player_array).flatten()
        top_ratings_indices = ratings.argsort()[-10:][::-1]
        recommended_player_ids = [
            self.player_encoded2player.get(players_not_watched[x][0]) for x in top_ratings_indices
        ]

        recommended_players = player_club_ratings[player_club_ratings["Player"].isin(recommended_player_ids)]
        recommended_players = recommended_players.drop_duplicates(subset=["player"])
        recommended_players = recommended_players.sort_values(by=['rating'], ascending=False)

        return recommended_players
