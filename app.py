import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Load the preprocessed data and model
with open(r'F:\\VIDEO_RECOMMODATION_SYSTEM\\artifacts\\movie_list.pkl', 'rb') as file:
    data = pickle.load(file)

# Convert tags to TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(data['tags'].astype(str))

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Collaborative filtering preparation
C = data['vote_average'].mean()
m = data['vote_count'].quantile(0.70)
qualified = data[data['vote_count'] >= m].copy()
qualified['weighted_rating'] = qualified.apply(
    lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) +
              (m / (m + x['vote_count']) * C), axis=1)

# Hybrid recommendation function
def hybrid_recommendation(title):
    idx = data[data['title'] == title].index[0]
    content_scores = list(enumerate(cosine_sim[idx]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)

    max_wr = qualified['weighted_rating'].max()
    qualified['normalized_wr'] = qualified['weighted_rating'] / max_wr

    combined_scores = []
    for i, c_score in content_scores:
        content_similarity = c_score if not isinstance(c_score, (list, np.ndarray)) else c_score[0] if c_score else 0
        title_ = data.iloc[i]['title']
        homepage = data.iloc[i]['homepage'] if 'homepage' in data.columns else None
        if title_ in qualified['title'].values:
            collab_score = qualified[qualified['title'] == title_]['normalized_wr'].values[0]
            rating = int(round(qualified[qualified['title'] == title_]['weighted_rating'].values[0]))
        else:
            collab_score, rating = 0, 0
        combined_scores.append((title_, homepage, rating, 0.5 * content_similarity + 0.5 * collab_score))

    combined_scores = sorted(combined_scores, key=lambda x: x[3], reverse=True)[1:11]
    recommendations = pd.DataFrame(combined_scores, columns=['Recommended Movie', 'Homepage', 'Rating', 'Combined Score'])
    return recommendations[['Recommended Movie', 'Homepage', 'Rating']]

# Streamlit UI with custom background
st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f0f4f8;
        background-image: linear-gradient(to bottom right, #ffecd2, #fcb69f);
        padding: 20px;
    }
    .recommendation-card {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s;
    }
    .recommendation-card:hover {
        transform: scale(1.02);
    }
    h4, p {
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✨ Movie Recommendation System 🎬")
st.subheader("Find movies similar to your favorite ones, with ratings and homepages beautifully displayed!")

movie_list = data['title'].values
selected_movie = st.selectbox("🎥 Select a movie to get recommendations:", movie_list)

if st.button("🔍 Recommend"):
    st.write(f"## ⭐ Top Recommendations for: {selected_movie}")
    recommendations = hybrid_recommendation(selected_movie)
    
    cols = st.columns(2)
    for index, row in recommendations.iterrows():
        col = cols[index % 2]
        with col:
            st.markdown(f"""
                <div class='recommendation-card'>
                    <h4>🎞️ {row['Recommended Movie']}</h4>
                    <p><strong>🌐 Homepage:</strong> <a href='{row['Homepage']}' target='_blank'>{row['Homepage']}</a></p>
                    <p><strong>⭐ Rating:</strong> {row['Rating']}</p>
                </div>
            """, unsafe_allow_html=True)
