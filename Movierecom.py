import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# laoding the dataset 
movies = pd.read_csv('movies.csv')

# convert the comma sep list of genre into a list of genre
movies['Genre_list'] = movies['Genre'].apply(
    lambda x:[g.strip() for g in x.split(',')] if isinstance(x,str)else []


)
# isinstance for checking if the the genre is there or not

mlb=MultiLabelBinarizer()

genre_dummies=pd.DataFrame(mlb.fit_transform(movies['Genre_list']),
                           columns=mlb.classes_,
                           index=movies.index)
''' genre_dummies will be like this
          Action  Comedy  Drama  Romance  Thriller
Movie A       1       1      0        0         0
Movie B       0       0      1        1         0
Movie C       1       0      0        0         1
Movie D       0       1      0        1         0
'''

director_dummies = pd.get_dummies(movies['Director'], prefix='Director')
star1_dummies    = pd.get_dummies(movies['Star1'], prefix='Star1')
star2_dummies    = pd.get_dummies(movies['Star2'], prefix='Star2')
rating = movies[['IMDB_Rating']]
features = pd.concat([genre_dummies, director_dummies, star1_dummies, star2_dummies, rating], axis=1)
def recommend_movies(liked,disliked,movies_df,features_df,top=12):
    liked_features =features_df[movies_df['Series_Title'].isin(liked)]
    disliked_features =features_df[movies_df['Series_Title'].isin(disliked)]
    if liked_features.empty:
        raise ValueError("Please provide at least one liked movie that exists in the dataset.")
    user_profile = liked_features.mean(axis=0)
    if not disliked_features.empty:
        user_profile -= disliked_features.mean(axis=0)
    #The user_profile (which is a pandas Series) is reshaped into a 2D array.  cosine_similarity expects 2D arrays as input.
    user_profile_vector = user_profile.values.reshape(1, -1)
    similarity_scores = cosine_similarity(features_df, user_profile_vector).flatten()
    movies_df=movies_df.copy() 
    movies_df['similarity']=similarity_scores
    movies_unseen = movies_df[~movies_df['Series_Title'].isin(liked + disliked)]
    recommendations= movies_unseen.sort_values(by='similarity',ascending=False).head(top)
    return recommendations[['Series_Title', 'similarity']]
    
liked_movies = ['Interstellar','The Dark Knight', 'Inception','Memento','2001: A Space Odyssey']
disliked_movies = ['Avengers: Infinity War', 'Avengers: Infinity War','Captain America: The Winter Soldier','The Wolf of Wall Street']

recommended=recommend_movies(liked_movies,disliked_movies,movies,features, top=10)

print("Top 10 Movie Recommendations:")
print(recommended)
