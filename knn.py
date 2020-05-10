# knn
import numpy as np
import pandas as pd
import heapq
from collections import Counter
from PIL import Image
import requests

# Get data
movie_data = pd.read_pickle('MoviePosters.csv').transpose()

# Get input
# url = input('Enter url to get rating: ')
url = "https://images-na.ssl-images-amazon.com/images/I/71niXI3lxlL._AC_SY679_.jpg"
input = Image.open(requests.get(url,stream=True).raw)
input = np.asarray(input.resize((32,32))).flatten() # resize the input image to 32*32*3 flattened

# Compute NN
nodes = [(np.linalg.norm(input - data), i) for i, data in enumerate(movie_data.posterData)]
heapq.heapify(nodes)
nearest = heapq.nsmallest(10, nodes)

mode = Counter()
for i in nearest:
    mode.update(round(movie_data['rating'][i[1]]))

rating = mode.most_common(1)
print(rating)
