import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridRecommender:
    def __init__(self, movie_df):
        self.movie_df = movie_df
        self._prepare()

    def _prepare(self):
        self.movie_df['genres'] = self.movie_df['genres'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movie_df['genres'])
        self.similarity = cosine_similarity(self.tfidf_matrix)

    def recommend(self, movie_title, top_n=5):
        indices = pd.Series(self.movie_df.index, index=self.movie_df['title']).drop_duplicates()
        idx = indices.get(movie_title, None)
        if idx is None:
            return []
        sim_scores = list(enumerate(self.similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movie_df['title'].iloc[movie_indices].tolist()
