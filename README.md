# USING MOVIE PLOT FOR GENRE CLASSIFICATION
## Overview
This project uses summarized plots of movies for classifying their genre. TF-IDF and BERT have been used to generate two different sets of embeddings that have been use in modelling.

## Table of Contents
- Overview
- Table of Contents
- Dataset
- Preprocessing
- Training
- Evaluation
- Usage
- Dependencies

## Dataset
The dataset used here has been taken from kaggle, you can get it [here](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots). It contained details about ~35,0000 movies including plot summaries, release year, origin, etc.

## Preprocessing

The movie genre column had 2265 unique values, including repetitions, different spellings and other redundancies.<br><br>

All of these were condensed down to 11 major movie genres with enough data for classification purpose. The list for them is given below in order number of observation for each of them:<br>

| Genre | No. of Movies |
|-------|---------------|
| Horror | 1167 |
| Action | 1098 |
| Thriller | 966 |
| Romance | 923 |
| Western | 865 |
| Science Fiction | 639 |
| Crime | 568 |
| Adventure | 526 |
| Musical | 467 |
| Film Noir | 345 |
| Mystery | 310 |
| **TOTAL** | 7874 |

Afterwards the genres are encoded using label encoder and then embeddings are created using __TF-IDF Vectorizer__ and __BERT Base uncased__ model. Afterwards seperate models were built for both set of embeddings.

## Training

### Initial Modelling

Embeddings are directly used to train and test models to test models to set a quick initial baseline performance.

### Using Class weights

Since the number of observations of different classes vary by a wide margin class weights are computed to help models learn better.