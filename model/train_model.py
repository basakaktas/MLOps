import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib


ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")


user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)


knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_movie_matrix.T)


joblib.dump(knn, "knn_model.pkl")
user_movie_matrix.T.to_pickle("movie_matrix.pkl")
