import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

tweet = 'Skillcate is a great Youtube Channel to learn Data Science'
print(SentimentIntensityAnalyzer().polarity_scores(tweet))
