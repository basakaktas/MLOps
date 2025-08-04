from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

knn = joblib.load("../model/knn_model.pkl")
movie_matrix = pd.read_pickle("../model/movie_matrix.pkl")
movies_df = pd.read_csv("../model/movies.csv")

@app.get("/predict/")
def recommend(movie_title: str):
    
    try:
        movie_id = movies_df[movies_df['title'].str.contains(movie_title, case=False)].iloc[0]['movieId']
    except IndexError:
        raise HTTPException(status_code=404, detail="Movie not found")

    try:
        movie_idx = list(movie_matrix.columns).index(movie_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Movie ID not in matrix")

    distances, indices = knn.kneighbors([movie_matrix.iloc[:, movie_idx]], n_neighbors=6)

    recommendations = []
    for i in range(1, len(indices[0])):
        rec_movie_id = movie_matrix.columns[indices[0][i]]
        title = movies_df[movies_df["movieId"] == rec_movie_id]["title"].values[0]
        recommendations.append(title)

    return {"recommendations": recommendations}
