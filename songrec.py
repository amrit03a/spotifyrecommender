import pandas as pd
import numpy as np
import pickle
import faiss
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

with open("songs_data.pkl", "rb") as f:
    music, normed_dense = pickle.load(f)

index = faiss.read_index("faiss_index.idx")
CLIENT_ID = "2be3e61b09bd47ea8b3687ee560f0ff5"
CLIENT_SECRET = "b33633e2829e4a918decbcddceed0bd2"

client_credentials_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        return album_cover_url
    else:
        # fallback image if not found
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def recommend(name):
    recommended_list = []
    recommended_music_posters = []

    idx = music[music['song'] == name].index[0]

    vector = normed_dense[idx].reshape(1, -1)
    distances, indices = index.search(vector, k=6)
    for i in indices[0][1:]:
        song_name = music.iloc[i]['song']
        tags = music.iloc[i]['tags']
        artist = tags.split()[0]

        recommended_music_posters.append(get_song_album_cover_url(song_name, artist))
        recommended_list.append(song_name)

    return recommended_list, recommended_music_posters

st.header('🎵 Music Recommender System (FAISS Powered)')

music_list = music['song'].values
selected_song = st.selectbox(
    "Type or select a song from the dropdown",
    music_list
)

if st.button('Show Recommendation'):
    recommended_music_names, recommended_music_posters = recommend(selected_song)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommended_music_names[0])
        st.image(recommended_music_posters[0])

    with col2:
        st.text(recommended_music_names[1])
        st.image(recommended_music_posters[1])

    with col3:
        st.text(recommended_music_names[2])
        st.image(recommended_music_posters[2])

    with col4:
        st.text(recommended_music_names[3])
        st.image(recommended_music_posters[3])

    with col5:
        st.text(recommended_music_names[4])
        st.image(recommended_music_posters[4])
