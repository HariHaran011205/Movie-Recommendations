from flask import Flask, request, jsonify
import pandas as pd
from models.hybrid_model import HybridRecommender

app = Flask(__name__)
movie_df = pd.read_csv('data/movies.csv')
recommender = HybridRecommender(movie_df)

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    recommendations = recommender.recommend(title)
    return jsonify({'recommendations': recommendations})
