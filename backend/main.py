import os
import pandas as pd
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    movie_matrix = pd.read_pickle(os.path.join(BASE_DIR, "movie_matrix.pkl"))
    with open(os.path.join(BASE_DIR, "knn_model.pkl"), "rb") as f:
        knn = pickle.load(f)
    movies = pd.read_csv(os.path.join(BASE_DIR, "movies.csv"))
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}\n"
                     f"Current directory: {BASE_DIR}\n"
                     f"Files expected: {os.listdir(BASE_DIR)}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Movie Recommendation API",
        "endpoints": {
            "root": "/",
            "predict": "/predict/?movie_title={title}",
            "health": "/health"
        }
    }

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

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8080))  
    uvicorn.run("main:app", host="0.0.0.0", port=port)

