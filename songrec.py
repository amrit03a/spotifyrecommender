import os
os.environ["HF_HOME"] = "/tmp/huggingface"

import os
import pickle
import faiss
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


# === Load dataset & FAISS (cached) ===
@st.cache_resource
def load_data():
    songs_path = hf_hub_download(
        repo_id="amrit0305/spotify",
        filename="songs_data.pkl",
        repo_type="dataset",
    )

    faiss_path = hf_hub_download(
        repo_id="amrit0305/spotify",
        filename="faiss_index.idx",
        repo_type="dataset",
    )

    with open(songs_path, "rb") as f:
        songs_data = pickle.load(f)

    # Handle different pickle formats
    if isinstance(songs_data, tuple):
        music, features = songs_data
        normed_dense = np.array(features).astype("float32")
    elif isinstance(songs_data, list):
        music = pd.DataFrame(songs_data)
        normed_dense = np.array(list(music["features"])).astype("float32")
    elif isinstance(songs_data, pd.DataFrame):
        music = songs_data
        normed_dense = np.array(list(music["features"])).astype("float32")
    else:
        raise ValueError("Unsupported songs_data format")

    faiss_index = faiss.read_index(faiss_path)

    return music, normed_dense, faiss_index


# === Spotify API (cached client) ===
@st.cache_resource
def get_spotify_client():
    CLIENT_ID = "2be3e61b09bd47ea8b3687ee560f0ff5"
    CLIENT_SECRET = "b33633e2829e4a918decbcddceed0bd2"
    return spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=CLIENT_ID, client_secret=CLIENT_SECRET
        )
    )


music, normed_dense, index = load_data()
sp = get_spotify_client()


# === Album Cover Lookup (cached) ===
@st.cache_data(show_spinner=False)
def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track", limit=1)
    if results and results["tracks"]["items"]:
        return results["tracks"]["items"][0]["album"]["images"][0]["url"]
    return "https://i.postimg.cc/0QNxYz4V/social.png"  # fallback image


# === Recommendation Engine ===
def recommend(name):
    recommended_list = []
    recommended_music_posters = []

    idx = music[music["song"] == name].index[0]
    vector = normed_dense[idx].reshape(1, -1)
    distances, indices = index.search(vector, k=6)

    for i in indices[0][1:]:
        song_name = music.iloc[i]["song"]
        tags = music.iloc[i].get("tags", "")
        artist = tags.split()[0] if isinstance(tags, str) and tags else ""
        recommended_list.append(song_name)
        recommended_music_posters.append(get_song_album_cover_url(song_name, artist))

    return recommended_list, recommended_music_posters


# === Streamlit UI ===
st.header("🎵 Music Recommender System (FAISS Powered)")
music_list = music["song"].values
selected_song = st.selectbox("Type or select a song from the dropdown", music_list)

if st.button("Show Recommendation"):
    recommended_music_names, recommended_music_posters = recommend(selected_song)
    cols = st.columns(5)
    for i in range(min(5, len(recommended_music_names))):
        with cols[i]:
            st.text(recommended_music_names[i])
            st.image(recommended_music_posters[i])
