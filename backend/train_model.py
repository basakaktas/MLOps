import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle


movies = pd.read_csv("../data/movies.csv")
ratings = pd.read_csv("../data/ratings.csv")


user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

user_movie_matrix.fillna(0, inplace=True)

movie_matrix = user_movie_matrix.T  

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(movie_matrix.values)


movie_matrix.to_pickle("movie_matrix.pkl")
with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)
movies.to_csv("movies.csv", index=False)
