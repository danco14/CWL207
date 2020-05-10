from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import heapq
from collections import Counter
from PIL import Image
import requests
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

app = Flask(__name__)
movie_data = pd.read_pickle('MoviePosters.csv').transpose()

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['poster_url']
    try:
        input = Image.open(requests.get(url,stream=True).raw)
    except:
        return '<div style="text-align:center"> <h1>Rating Not Available. Image URL could not be accessed. Please try a new poster. </h1><br/> <a style="font-size:30" href="/">Go Back Home</a> </div>'
    input = np.asarray(input.resize((32,32))).flatten() # resize the input image to 32*32*3 flattened

    # Compute NN
    nodes = [(np.linalg.norm(input - data), i) if data.shape[0] == 3072 else (float('inf'),i) for i, data in enumerate(movie_data.posterData) ]
    heapq.heapify(nodes)
    nearest = heapq.nsmallest(30, nodes)

    closest = []
    for i in range(30):
        closest.append(movie_data.iloc[nearest[i][1]])
    closest = pd.concat(closest,axis=1).transpose()

    mode = Counter()
    unrounded = {}
    for i in nearest:
        rounded = round(movie_data['rating'][i[1]])
        mode.update([rounded])
        if (rounded not in unrounded):
            unrounded[rounded]=movie_data['rating'][i[1]]
        else:
            unrounded[rounded]+=movie_data['rating'][i[1]]

    rating = mode.most_common(1)[0][0]
    predicted_rating = round(unrounded[rating]/mode.most_common(1)[0][1],1)

    img = io.BytesIO()
    sns.set()
    plt.clf()
    plt.cla()
    plt.hist(closest.rating,bins=20)
    plt.xlabel('Ratings')
    plt.ylabel('Count')
    plt.title('Histogram of 30 closest movie posters')
    plt.savefig(img,format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('predict.html',postersrc=url,rating=predicted_rating,plot_url=plot_url)
