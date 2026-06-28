import streamlit as st
import pickle
import pandas as pd
movies_list=pickle.load(open('movie.pickle','rb'))
movies_list=movies_list['title'].values
st.title('Movie recommender system')
options=st.selectbox('list of movies',movies_list)

