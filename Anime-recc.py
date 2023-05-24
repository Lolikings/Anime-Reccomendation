import pandas as pd
anime = pd.read_csv("anime.csv")
print(anime.shape)
print(anime.columns)
anime['Score-10'].head(10)

# %%
#anime.isnull().sum()

# %%
#Popularty base Reccomendation system
anime = anime.drop(anime[anime['Score'] == 'Unknown'].index)
anime.sort_values('Score',ascending=False).head(50)




# %%
x = anime.groupby('Members').count()['Score']>0
u = x[x].index
print(x)


# %%
filter = anime[anime['Members'].isin(u)]
print(filter)


# %%
y = filter.groupby('Name').count()['Score']>0
f_a = y[y].index

# %%
f_s = filter[filter['Name'].isin(f_a)]
print(f_s)

# %%
pt = filter.pivot_table(index='Name',columns='Members',values='Score')
pt.fillna(0,inplace=True)
pt

# %%
from sklearn.metrics.pairwise import cosine_similarity
simi = cosine_similarity(pt)
simi.shape
simi

# %%
import numpy as n 
import random
def recc(anime_name):
    if anime_name not in pt.index:
        print("Anime not found in the dataset.")
        return []
    
    index = n.where(pt.index == anime_name)[0][0]
    recc = sorted(list(enumerate(simi[index])), key=lambda x: x[1], reverse=True)[1:5]
    random.shuffle(recc)

    d = []
    for i in recc:
        animes = []
        s = anime[anime['Name'] == pt.index[i[0]]]
        animes.extend(list(s['Name'].values))
        animes.extend(list(s['Score'].values))
        d.append(animes)
    
    return d




# %%
recc('Naruto')

# %%
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

x = anime.groupby('Members').count()['Score'] > 0
u = x[x].index
filtered_anime = anime[anime['Members'].isin(u)]

y = filtered_anime.groupby('Name').count()['Score'] > 0
famous_anime = y[y].index
filtered_scores = filtered_anime[filtered_anime['Name'].isin(famous_anime)]

pt = filtered_scores.pivot_table(index='Name', columns='Members', values='Score')
pt.fillna(0, inplace=True)

similarity_matrix = cosine_similarity(pt)



# %%
def recc(anime_name):
    if anime_name not in pt.index:
        print("Anime not found in the dataset.")
        return []
    
    index = np.where(pt.index == anime_name)[0][0]
    recommendations = list(enumerate(similarity_matrix[index]))
    print(len(recommendations))
    random.shuffle(recommendations)

    recommended_anime = []
    for i in recommendations[:5]:
        similar_anime_index = i[0]
        similar_anime_name = pt.index[similar_anime_index]
        recommended_anime.append(similar_anime_name)
    
    return recommended_anime

recc('One Piece')


# %%
from surprise import Dataset, Reader, SVD

# Load the anime ratings dataset
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(anime[['Members', 'Name', 'Score']], reader)

# Build the matrix factorization model (SVD)
model = SVD()
model.fit(data.build_full_trainset())

def recc(anime_name):
    try:
        # Predict the ratings for the given anime for all users
        predictions = []
        for user_id in model.trainset.all_users():
            prediction = model.predict(user_id, anime_name)
            predictions.append((user_id, prediction.est))
        
        # Sort the predictions in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get the top 5 recommendations
        recommended_anime = [model.trainset.to_raw_iid(pred[0]) for pred in predictions[:5]]
        
        return recommended_anime
    except ValueError:
        print("Anime not found in the dataset.")
        return []

recc('One Piece')
