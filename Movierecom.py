import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

df = pd.read_csv('movies.csv')

# Convert genre string to list
df['genres'] = df['Genre'].apply(lambda x: [g.strip() for g in x.split(',')] if isinstance(x, str) else [])

# One-hot encode genres
mlb = MultiLabelBinarizer()
genre = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_, index=df.index)
dir=pd.get_dummies(df['Director'], prefix='Dir')
s1= pd.get_dummies(df['Star1'], prefix='S1')
s2=pd.get_dummies(df['Star2'], prefix='S2')
scaler = MinMaxScaler()
rat = pd.DataFrame(scaler.fit_transform(df[['IMDB_Rating']]), columns=['IMDB_Rating'], index=df.index)
X = pd.concat([genre, dir, s1, s2, rat], axis=1)

def recommend(liked, disliked, df, X, top_n=10):
    liked=X[df['Series_Title'].isin(liked)]
    disliked=X[df['Series_Title'].isin(disliked)]
    
    if liked.empty:
        raise ValueError("Liked list is empty or doesn't match any movies.")
    
    profile = liked.mean()
    if not disliked.empty:
        profile -= disliked.mean()
    
    sim = cosine_similarity(X, profile.values.reshape(1, -1)).flatten()
    df = df.copy()
    df['sim'] = sim
    
    unseen=df[~df['Series_Title'].isin(liked + disliked)]
    recs=unseen.sort_values(by='sim',ascending=False).head(top_n)
    
    return recs[['Series_Title','sim']]

# Test 
dislikedd = ['Interstellar', 'Gladiator', 'Inception', 'Saving Private Ryan']
likedd = ['Spider-Man: Into the Spider-Verse', 'Star Wars: Episode VI - Return of the Jedi', 'Toy Story']

top=recommend(likedd, dislikedd, df, X, top_n=10)

print("Top 10 Movie Recommendations:")
print(top)
