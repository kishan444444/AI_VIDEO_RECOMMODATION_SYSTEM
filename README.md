# 🎬 Movie Recommendation System

A **hybrid movie recommender** built with **Streamlit**, combining **content-based filtering** (TF-IDF + cosine similarity) and **collaborative filtering** (IMDb-style weighted ratings). Get intelligent, visually engaging movie recommendations based on your favorites.

---

## 🚀 Features

* 🔍 **Search by Movie Title** to discover similar movies
* 🧠 **TF-IDF Vectorization** on tags for semantic similarity
* 📊 **Weighted Rating System** based on votes and averages
* �퇕 **Hybrid Scoring** blending content and collaborative approaches
* 🌐 **Homepage Links** to explore more about recommended movies
* 🎨 **Stylish UI** with gradient backgrounds and responsive cards

---

## 🖼️ Preview

| Homepage                                                       | Recommendations                                                    |
| -------------------------------------------------------------- | ------------------------------------------------------------------ |
| ![Home](https://via.placeholder.com/400x250.png?text=App+Home) | ![Cards](https://via.placeholder.com/400x250.png?text=Movie+Cards) |

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Streamlit**
* **scikit-learn**
* **pandas**, **numpy**
* **pickle** (for data loading)
* **TfidfVectorizer** (for feature extraction)

---

## ⚙️ Installation & Run

```bash
# Clone the repo
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🧠 How It Works

1. User selects a movie title.
2. Tags are transformed into TF-IDF vectors.
3. Cosine similarity finds semantically similar movies.
4. Vote count and average compute a weighted rating.
5. Combined scores rank top recommendations.
6. Streamlit UI displays titles, homepages, and ratings.

---

## 📁 Project Structure

```
├── app.py                      # Main Streamlit application
├── artifacts/
│   └── movie_list.pkl          # Preprocessed movie data
├── requirements.txt
└── README.md
```

---

## 📝 Example Use Case

> "I like *The Dark Knight* — what else should I watch?"
>
> The system suggests high-rated, thematically similar titles like *Inception*, *Joker*, and *Batman Begins* with homepage links and ratings.

---



---
