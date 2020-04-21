# Reddit Flair Detector

A Reddit Flair Detector web application to detect flairs of India subreddit posts using Machine Learning algorithms. The application can be found live at [Reddit Flair Detector](https://redditindiaflair.herokuapp.com).

### Structure

#### [Data Scrape Notebook](https://github.com/manalarora/reddit-flair-detection/blob/master/DataScrape.ipynb) 
contains code used to scrape data off many posts from reddit using praw api and save all that data in a csv file. 
***goto notebook to know more***
 
#### [Data Models Notebook](https://github.com/manalarora/reddit-flair-detection/blob/master/DataModels.ipynb) 
contains code used to do the data analysis and train various machine learning models and check the accuracy on different features. The models giving best results were downloaded.
##### Different Features are :-
* ***Title*** - Title of the post
* ***Url*** - URL of the post
* ***Body*** - Body of the post
* ***Comments*** - Comments of the post 
* ***All features combined*** - Combination of all the above mentioned features

##### Different Models are :-
* ***Naive Bayes Classifier*** 
* ***Linear SVM*** - Usually converged after 5-7 iterations 
* ***Logistic Regression*** - Usually converged after 100-150 iterations 
* ***Random Forrest*** - Showed the best results at most times 
* ***Basic Neural Network*** - Tried varioius approaches adjusting number of layers ,number of neurons per layer and number of iterations but did not show good results  
***goto notebook to know more*** 

#### [Data Directory](https://github.com/manalarora/reddit-flair-detection/tree/master/data) 
the directory contains 4 different csv files containg data that was scraped using Data Scrape notebook
* *data1.csv*  
    * Initial Data Scraping of 100 posts per category
* *data2.csv* 
    * Data Scraping of categories done individually so that i can create a dataset with zero empty values
    * Done because their was very poor performance on flair prediction using body as a feature so I thought that was because of some posts not having body in the dataset
* *data3.csv* 
    * Data scraping of 100 posts per category but redifing combined_features 
    * after data2 their was minimal improvement in body so I thought that body is not a useful feature thats why I excluded body from combined features
* *data4.csv* 
    * similar to data3 it just has 150 posts per category to train on a larger dataset 

#### [Models Directory](https://github.com/manalarora/reddit-flair-detection/tree/master/models) 
contains best performing models trained on all data.csv files. Random Forrest models could not be uploaded due to their large size

#### [Website Directory](https://github.com/manalarora/reddit-flair-detection/tree/master/website) 
contains the flask implementation of the app. It takes in a url for reddit india post and gives the prediction according to the trained models

##### Running the website locally 
Create a virtual environment, install the dependencies, start the server.
```sh
$ virtualenv -p python3 env
$ source env/bin/activate
$ pip install -r requirements.txt
$ python3 app.py
```

### Results

#### Title as Feature
| Machine Learning Algorithm | Accuracy_data1.csv | Accuracy_data2.csv | Accuracy_data3.csv | Accuracy_data4.csv |
| ------ | ------ | ------ | ------ | ------ |
| Naive Bayes | 0.65 | 0.63 | 0.67 | 0.65 |
| Linear SVM | 0.71 | 0.65 | 0.72 | 0.69 | 
| Logistic Regression | 0.72 | 0.68 | 0.73 | 0.69 |
| Random Forest | 0.70 | 0.66 | 0.74 | 0.69 |
| MLP | 0.47 | 0.48 | 0.54 | 0.52 |

#### Body as Feature
| Machine Learning Algorithm | Accuracy_data1.csv | Accuracy_data2.csv | Accuracy_data3.csv | Accuracy_data4.csv |
| ------ | ------ | ------ | ------ | ------ |
| Naive Bayes | 0.25 | 0.26 | 0.24 | 0.27 |
| Linear SVM | 0.38 | 0.35 | 0.35 | 0.39 | 
| Logistic Regression | 0.37 | 0.33 | 0.29 | 0.36 |
| Random Forest | 0.35 | 0.35 | 0.40 | 0.37 |
| MLP | 0.28 | 0.26 | 0.24 | 0.27 |

#### Comments as Feature
| Machine Learning Algorithm | Accuracy_data1.csv | Accuracy_data2.csv | Accuracy_data3.csv | Accuracy_data4.csv |
| ------ | ------ | ------ | ------ | ------ |
| Naive Bayes | 0.36 | 0.34 | 0.32 | 0.36 |
| Linear SVM | 0.35 | 0.40 | 0.40 | 0.43 | 
| Logistic Regression | 0.38 | 0.41 | 0.37 | 0.44 |
| Random Forest | 0.38 | 0.37 | 0.40 | 0.44 |
| MLP | 0.25 | 0.32 | 0.35 | 0.39 |

#### Combined features(Title + Comments + Body[data1 and dadta2]) as Feature
| Machine Learning Algorithm | Accuracy_data1.csv | Accuracy_data2.csv | Accuracy_data3.csv | Accuracy_data4.csv |
| ------ | ------ | ------ | ------ | ------ |
| Naive Bayes | 0.53 | 0.53 | 0.55 | 0.53 |
| Linear SVM | 0.72 | 0.71 | 0.68 | 0.68 | 
| Logistic Regression | 0.75 | 0.75 | 0.68 | 0.71 |
| Random Forest | 0.78 | 0.78 | 0.72 | 0.70 |
| MLP | 0.47 | 0.52 | 0.40 | 0.41 |

### References

#### 1. For data scraping from reddit:
1. http://www.storybench.org/how-to-scrape-reddit-with-python/
2. https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
3. https://praw.readthedocs.io/en/latest/

#### 2. For applying various machine learning model:
1. https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
2. https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#building-a-pipeline
3. https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/


#### 3.For Deploying the model on heroku:
1. https://towardsdatascience.com/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b

