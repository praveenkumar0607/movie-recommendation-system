import os
import requests
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings  # To access the settings

# Define the path to the CSV file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'recommendation', 'media', 'dataset.csv')

# Load your data and model from the notebook
movies_df = pd.read_csv(CSV_PATH)
# Ensure the 'combined_features' column exists or create it if needed
if 'combined_features' not in movies_df.columns:
    # Example: create combined features from existing columns (adjust as necessary)
    movies_df['combined_features'] = movies_df.apply(lambda row: ' '.join(row.astype(str)), axis=1)

count_matrix = CountVectorizer().fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(count_matrix)

# Fetch the OMDb API key from settings
OMDB_API_KEY = settings.OMDB_API_KEY

def fetch_poster_url(title):
    url = f'http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}'
    response = requests.get(url)
    data = response.json()
    return data.get('Poster', '')

# Function to get recommendations
def get_recommendations(title):
    try:
        idx = movies_df[movies_df['title'] == title].index[0]
    except IndexError:
        return []  # If the movie is not found, return an empty list
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get the scores of the 10 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    
    recommendations = movies_df.iloc[movie_indices][['title']].to_dict(orient='records')
    for rec in recommendations:
        rec['poster_url'] = fetch_poster_url(rec['title'])
    
    return recommendations

def get_top_20_movies():
    top_20_movies = movies_df.head(20)[['title']]
    top_20_movies['poster_url'] = top_20_movies['title'].apply(fetch_poster_url)
    return top_20_movies.to_dict(orient='records')

def recommend_movies(request):
    context = {}
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest' and request.method == 'GET':
        term = request.GET.get('term', '')
        movies = movies_df[movies_df['title'].str.contains(term, case=False, na=False)]
        movie_titles = movies['title'].tolist()
        return JsonResponse(movie_titles, safe=False)
    
    if request.method == 'POST':
        movie_title = request.POST.get('movie_title')
        if movie_title:
            recommendations = get_recommendations(movie_title)
            context['recommendations'] = recommendations
        else:
            context['error'] = "Please select a movie."
        
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return render(request, 'recommendations_fragment.html', {'recommendations': recommendations})
    
    # On initial load, show top 20 movies
    context['top_movies'] = get_top_20_movies()
    
    return render(request, 'recommend.html', context)
