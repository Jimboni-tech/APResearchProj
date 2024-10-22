import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
text = "I really enjoyed this book!"
scores = analyzer.polarity_scores(text)
print(scores)


from textblob import TextBlob

text = "I really enjoyed this book!"
blob = TextBlob(text)
sentiment = blob.sentiment
print(sentiment) 