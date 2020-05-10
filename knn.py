# knn
import numpy as np
import pandas as pd
import heapq
from collections import Counter
from PIL import Image
import requests

# Get data
movie_data = pd.read_pickle('MoviePosters.csv').transpose()

# Keep asking for input until valid image url is passed
while True:
    try:
        url = input('Enter poster url to get rating: ')
        input = Image.open(requests.get(url,stream=True).raw)
        break
    except:
        print("\n\n Image couldn't be fetched, please try a different image. \n\n")

input = np.asarray(input.resize((32,32))).flatten() # resize the input image to 32*32*3 flattened

# Compute NN
nodes = [(np.linalg.norm(input - data), i) if data.shape[0] == 3072 else (float('inf'),i) for i, data in enumerate(movie_data.posterData) ]
heapq.heapify(nodes)
nearest = heapq.nsmallest(30, nodes)

closest = []
for i in range(3):
    closest.append(movie_data.iloc[nearest[i][1]])

mode = Counter()
unrounded = {}
for i in nearest:
    # img = Image.open(requests.get(movie_data['Poster'][i[1]], stream=True).raw)
    # img.show()
    rounded = round(movie_data['rating'][i[1]])
    mode.update([rounded])
    if (rounded not in unrounded):
        unrounded[rounded]=movie_data['rating'][i[1]]
    else:
        unrounded[rounded]+=movie_data['rating'][i[1]]

rating = mode.most_common(1)[0][0]
predicted_rating = round(unrounded[rating]/mode.most_common(1)[0][1],1)
print(predicted_rating)
