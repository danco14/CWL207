# CWL207

For this project, we borrowed a dataset (https://www.kaggle.com/neha1703/movie-genre-from-its-poster) of imdb movie listings and modified it so that we have a dataset of 10,000 movie posters along with their imdb ratings.
Using this, we created a tool that can take in a movie poster's image url and return an estimated rating of the movie based on what similar looking movie posters received. Our tool will return the predicted rating along with a histogram of the ratings of the similar movie posters.

In order to run, you will need to have python3 installed.
When in the project folder:
1) run "pip install -r requirements.txt" in order to install all the required libraries
2) export FLASK_APP=main.py
3) flask run
4) open up http://127.0.0.1:5000/ in a browser

if flask doesn't work check this out: https://flask.palletsprojects.com/en/1.1.x/quickstart/

![Starting page to enter the poster URL]('./homepage.png')
![Predicted Rating and Histogram]('./predict.png')
