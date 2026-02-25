import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load the dataset
movies = pd.read_csv("movie data.csv")
credits = pd.read_csv("credits data.csv")

#merge both dataset
movies = movies.merge(credits, on="title")

movies = movies[['title', 'overview', 'genres', 'keywords', 'cast', 'crew']] #select import columns from merge dataset

#fill missing value
movies['overview'] = movies['overview'].fillna("")

#
def convert(text):
    L = []
    if pd.isna(text):
        return L
    try:
        data = ast.literal_eval(text)
        for i in data:
            L.append(i['name'])
    except:
        return L
    return L

# apply conversion
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# take first 3 cast
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])

#
def fetch_director(text):
    L = []
    if pd.isna(text):
        return L
    try:
        data = ast.literal_eval(text)
        for i in data:
            if i['job'] == 'Director':
                L.append(i['name'])
    except:
        return L
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

#remove space from name
for col in ['genres', 'keywords', 'cast', 'crew']:
    movies[col] = movies[col].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

#
movies['tags'] = (
    movies['overview'] + " " +
    movies['genres'].astype(str) + " " +
    movies['keywords'].astype(str) + " " +
    movies['cast'].astype(str) + " " +
    movies['crew'].astype(str)
)

movies = movies[['title', 'tags']]
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

#
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

#
similarity = cosine_similarity(vectors)

#save model
pickle.dump(movies, open("movies.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("\n✅ Recommendation model trained & saved successfully!")