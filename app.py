import streamlit as st
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- Page Config ----------------------

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="centered"
)

# ---------------------- Load Data ----------------------

@st.cache_resource
def load_data():

    # Load processed dataframe
    movies = pickle.load(open("movies.pickle", "rb"))

    # Create vectors
    cv = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    vectors = cv.fit_transform(movies["tags"]).toarray()

    # Create similarity matrix
    similarity = cosine_similarity(vectors)

    return movies, similarity


movies, similarity = load_data()

# ---------------------- Recommendation Function ----------------------

def recommend(movie):

    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(
            movies.iloc[i[0]].title
        )

    return recommended_movies


# ---------------------- Streamlit UI ----------------------

st.title("🎬 Movie Recommender System")

st.write("Reached UI")

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)

st.write("Reached Selectbox")

st.button("Recommend")

st.write("Reached Button")