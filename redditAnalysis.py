import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

sentiment = pd.read_csv("Data/BitcoinSubmissions_Cleaned.csv")
analyzer = SentimentIntensityAnalyzer()
def get_sentiment_score(text):
    if isinstance(text, str):  # Check if the text is a string
        return analyzer.polarity_scores(text)['compound']
    
    return 0


sentiment['submission_sentiment'] = sentiment['cleaned_submission'].apply(get_sentiment_score)
sentiment_df = sentiment[['date', 'submission_sentiment']].copy()
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
sentiment_df = sentiment_df.groupby('date').mean().reset_index()
sentiment_df.to_csv('Data/sentiment_data.csv', index=False)


