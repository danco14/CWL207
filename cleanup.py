#clean up the data
import pandas as pd
import numpy as np
from PIL import Image
import requests

movies = pd.read_csv('MovieGenre.csv', engine='python')
movies.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
movies.drop(['Genre'],axis=1,inplace=True)
movies.rename(columns={'IMDB Score':'rating','imdbId':'movieId','Imdb Link':'link'},inplace=True)
movies_reduced = movies.iloc[::4] #reduce dataset size by 4

temp = []
for index,movie in movies_reduced.iterrows():
    try:
        poster = Image.open(requests.get(movie.Poster,stream=True).raw)
        poster = np.asarray(poster.resize((32,32))).flatten() # 32*32*3 image data
        mcopy = movie.copy()
        mcopy['posterData'] = poster
        temp.append(mcopy)
    except:
        pass

clean_data = pd.concat(temp,axis=1,ignore_index=True)
clean_data.to_pickle('MoviePosters.csv')


# read_clean = pd.read_pickle('MoviePosters.csv')
# print(read_clean.shape)
# print(read_clean[6018])
