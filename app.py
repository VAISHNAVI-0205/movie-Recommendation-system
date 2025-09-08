from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample movie dataset
movies = pd.DataFrame([
    {"movie_id": 1, "title": "Sabdham", "genre": "Horror Thriller",
     "description": "A chilling horror thriller with gripping sound design."},
    {"movie_id": 2, "title": "Kudumbasthan", "genre": "Comedy Drama",
     "description": "A family comedy drama filled with emotions and laughter."},
    {"movie_id": 3, "title": "Kadhalikka Neramillai", "genre": "Romantic Comedy",
     "description": "A light-hearted romantic comedy released for Pongal."},
    {"movie_id": 4, "title": "Madraskaaran", "genre": "Drama",
     "description": "A drama film set in Chennai, with music by AR Rahman."},
    {"movie_id": 5, "title": "Sweetheart!", "genre": "Romantic Comedy",
     "description": "A sweet romantic comedy starring Rio Raj and Gopika Ramesh."},
    {"movie_id": 6, "title": "Dragon", "genre": "Comedy Drama",
     "description": "A coming-of-age comedy drama with Pradeep Ranganathan."}
])

# Preprocess features
movies["features"] = movies["genre"] + " " + movies["description"]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(movies["features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movie(title, top_n=3):
    idx = movies[movies["title"].str.lower() == title.lower()].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][["title", "genre", "description"]]

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        movie_name = request.form["movie"]
        try:
            recommendations = recommend_movie(movie_name, top_n=3)
            return render_template("result.html", movie=movie_name, recs=recommendations.to_dict(orient="records"))
        except:
            return render_template("result.html", movie=movie_name, recs=None, error=True)
    return render_template("index.html", movies=movies["title"].tolist())

if __name__ == "__main__":
    app.run(debug=True)
