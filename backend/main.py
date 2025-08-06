import pandas as pd
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

movie_matrix = pd.read_pickle("movie_matrix.pkl")

with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

movies = pd.read_csv("movies.csv")


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/predict/")
def recommend(movie_title: str):
    try:
        movie_id = movies[movies['title'].str.contains(movie_title, case=False, na=False)].iloc[0]['movieId']
        movie_idx = movie_matrix.index.get_loc(movie_id)

        distances, indices = knn.kneighbors([movie_matrix.iloc[movie_idx].values], n_neighbors=6)

        similar_movies = []
        for i in indices.flatten()[1:]:
            similar_id = movie_matrix.index[i]
            title = movies[movies['movieId'] == similar_id]['title'].values[0]
            similar_movies.append(title)

        return {"recommendations": similar_movies}

    except Exception as e:
        return {"error": str(e)}
