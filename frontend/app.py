import streamlit as st
import requests

st.title("🎬 Movie Recommendation Engine")

movie = st.text_input("Enter a movie you like:", "Toy Story")

if st.button("Recommend"):
    with st.spinner("Finding recommendations..."):
        response = requests.get(f"http://localhost:8000/predict/", params={"movie_title": movie})
        if response.status_code == 200:
            recommendations = response.json()["recommendations"]
            st.success("Recommended Movies:")
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.error("Movie not found or error in prediction.")
