import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

mood_data = {
    "mood_text": [
        # Comedy
        "I want to laugh",
        "I'm in the mood for something funny",
        "I need a comedy",
        "Make me laugh",
        "I want something light-hearted",
        # Drama
        "I feel emotional",
        "I want a deep story",
        "I'm in the mood for drama",
        "I want something serious",
        "I want to watch a dramatic movie",
        # Romance
        "I feel romantic",
        "I want a love story",
        "I'm in the mood for romance",
        "I want to watch something about relationships",
        "I want to be loved",
        # Action
        "I want action",
        "I'm in the mood for a thrilling ride",
        "I want to watch fights and explosions",
        "I need something fast-paced",
        "I want an adrenaline rush",
        # Thriller
        "I want a thriller",
        "I'm craving suspense",
        "I want something suspenseful",
        "I want a psychological thriller",
        "I want to be on the edge of my seat",
        # Mystery
        "I want to solve a mystery",
        "I'm in the mood for a whodunit",
        "I want a detective story",
        "I want something mysterious",
        "I want to watch a mystery movie",
        # Horror
        "I want to be scared",
        "I'm in the mood for horror",
        "I want a scary movie",
        "I want to watch something creepy",
        "I want a horror film",
        # Fantasy
        "I want to escape reality",
        "I'm in the mood for fantasy",
        "I want to watch magic and adventure",
        "I want a fantasy world",
        "I want to see mythical creatures",
        # Sci-Fi
        "I want science fiction",
        "I'm in the mood for sci-fi",
        "I want to watch space adventures",
        "I want futuristic movies",
        "I want to see aliens and robots",
        # Animation
        "I want animation",
        "I'm in the mood for cartoons",
        "I want to watch an animated movie",
        "I want something for kids",
        "I want a family-friendly animation",
        # Family
        "I want a family movie",
        "I'm in the mood for something everyone can watch",
        "I want a wholesome movie",
        "I want to watch with my family",
        "I want a feel-good family film",
        # Biography
        "I want to watch a true story",
        "I'm in the mood for a biography",
        "I want to see real-life heroes",
        "I want a movie based on real events",
        "I want to watch someone's life story",
        # Musical
        "I want a musical",
        "I'm in the mood for music",
        "I want to watch people sing and dance",
        "I want a movie with great songs",
        "I want a musical film",
        # Adventure
        "I want an adventure",
        "I'm in the mood for exploring",
        "I want to watch a journey",
        "I want a movie with quests",
        "I want an adventurous film",
        # War
        "I want a war movie",
        "I'm in the mood for battles",
        "I want to watch soldiers",
        "I want a movie about war",
        "I want a historical war film",
        # History
        "I want a historical movie",
        "I'm in the mood for history",
        "I want to watch something from the past",
        "I want a movie based on history",
        "I want to learn about history",
    ],
    "genre": [
        "Comedy", "Comedy", "Comedy", "Comedy", "Comedy",
        "Drama", "Drama", "Drama", "Drama", "Drama",
        "Romance", "Romance", "Romance", "Romance", "Romance",
        "Action", "Action", "Action", "Action", "Action",
        "Thriller", "Thriller", "Thriller", "Thriller", "Thriller",
        "Mystery", "Mystery", "Mystery", "Mystery", "Mystery",
        "Horror", "Horror", "Horror", "Horror", "Horror",
        "Fantasy", "Fantasy", "Fantasy", "Fantasy", "Fantasy",
        "Sci-Fi", "Sci-Fi", "Sci-Fi", "Sci-Fi", "Sci-Fi",
        "Animation", "Animation", "Animation", "Animation", "Animation",
        "Family", "Family", "Family", "Family", "Family",
        "Biography", "Biography", "Biography", "Biography", "Biography",
        "Musical", "Musical", "Musical", "Musical", "Musical",
        "Adventure", "Adventure", "Adventure", "Adventure", "Adventure",
        "War", "War", "War", "War", "War",
        "History", "History", "History", "History", "History",
    ]
}
mm=min(len(mood_data["mood_text"]), len(mood_data["genre"]))
mood_data["mood_text"] =mood_data["mood_text"][:mm]
mood_data["genre"] =mood_data["genre"][:mm]
df_mood=pd.DataFrame(mood_data)
vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df_mood['mood_text'])
model=MultinomialNB()
model.fit(X, df_mood['genre'])
movies=pd.read_csv("movies.csv")
user_mood=input("How are you feeling today? → ")
moidelip = vectorizer.transform([user_mood])
pgrenre=model.predict(moidelip)[0]
print(f"\n🎯 You are in the mood for: {pgrenre} movies.\n")
matched_movies = movies[movies['Genre'].str.contains(pgrenre, case=False, na=False)]

if matched_movies.empty:
    print("No matching movies found.")
else:
    top_recs = matched_movies[['Series_Title', 'Genre', 'IMDB_Rating']].sample(n=5)
    print("🎬 Top Recommendations:\n")
    print(top_recs.to_string(index=False))
