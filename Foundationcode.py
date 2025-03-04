import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

with open(r'F:\VIDEO_RECOMMODATION_SYSTEM\artifacts\movie_list.pkl', 'rb') as file:
    data= pickle.load(file)
    
# Convert tags to TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english',max_features=5000)
tfidf_matrix = tfidf.fit_transform(data['tags'].astype(str))

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

C = data['vote_average'].mean()
m = data['vote_count'].quantile(0.70)
qualified = data[data['vote_count'] >= m].copy()

qualified['weighted_rating'] = qualified.apply(
    lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) +
              (m / (m + x['vote_count']) * C), axis=1)

def hybrid_recommendation(title, alpha=0.5):
    # Content-based score
    idx = data[data['title'] == title].index[0]
    content_scores = list(enumerate(cosine_sim[idx]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)

    # Collaborative score (normalized weighted_rating)
    max_wr = qualified['weighted_rating'].max()
    qualified['normalized_wr'] = qualified['weighted_rating'] / max_wr

    # Combine scores
    combined_scores = []
    for i, c_score in content_scores:
        if isinstance(c_score, (list, np.ndarray)):
            content_similarity = c_score[0] if len(c_score) > 0 else 0
        else:
            content_similarity = c_score

        title_ = data.iloc[i]['title']
        homepage = data.iloc[i]['homepage'] if 'homepage' in data.columns else None
        if title_ in qualified['title'].values:
            collab_score = qualified[qualified['title'] == title_]['normalized_wr'].values[0]
            rating = int(round(qualified[qualified['title'] == title_]['weighted_rating'].values[0]))
        else:
            collab_score = 0
            rating = 0
        combined_scores.append((title_, homepage, rating, alpha * content_similarity + (1 - alpha) * collab_score))

    # Top 10 recommendations
    combined_scores = sorted(combined_scores, key=lambda x: x[3], reverse=True)[1:11]

    # Convert to DataFrame for clean output with updated column names
    recommendations = pd.DataFrame(combined_scores, columns=['Recommended Movie', 'Homepage', 'Rating', 'Combined Score'])
    return recommendations[['Recommended Movie', 'Homepage', 'Rating']]

# Example usage
print(hybrid_recommendation("Inception"))



