# Data Collection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def movieRecommendation():
    moviedata = pd.read_csv("movies.csv")
    df = moviedata
    print(df)

    print(df.isnull().sum())
    choosed_column = ["genres","keywords","tagline","cast","director",'title']
    df = df[choosed_column]
    print(df.shape)
    for i in choosed_column:
        if df[i].isnull().sum() > 0:
            df[i] = df[i].fillna('')
    print(df.isnull().sum())

    combine_feature = ''
    for i in choosed_column:
        combine_feature = combine_feature+' '+df[i]
    print(combine_feature)

    vectorizer = TfidfVectorizer()
    feature_vector = vectorizer.fit_transform(combine_feature)
    print(feature_vector)
#     Consine Similarity
    similarity = cosine_similarity(feature_vector)


    # getting the movie name
    movie_name = input("Enter your Favorite Movie name\n")
    creatingListofMovies = df['title'].tolist()
    print(creatingListofMovies)

    find_close_match = difflib.get_close_matches(movie_name,creatingListofMovies)

    print(find_close_match[0])
    index_of_movie = moviedata[moviedata.title == find_close_match[0]]['index'].values[0]
    print(index_of_movie)
    similarityscore = list(enumerate(similarity[index_of_movie]))
    print(similarityscore)
    sortedSimiliarMovie =sorted(similarityscore,key = lambda x:x[1],reverse=True)
    print(sortedSimiliarMovie)

    i = 1
    for movie in sortedSimiliarMovie:
        index = movie[0]
        titleofMovie = moviedata[moviedata.index == index]['title'].values[0]
        if i<30:
            print(titleofMovie)
            i = i+1


if __name__=="__main__":
    movieRecommendation()