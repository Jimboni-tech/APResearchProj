import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Load Reddit data from CSV
reddit_df = pd.read_csv('Data/BitcoinSubmissions.csv')

reddit_df['date'] = pd.to_datetime(reddit_df['date'])


analyzer = SentimentIntensityAnalyzer()

# Function to calculate sentiment score
'''def get_sentiment_score(text):
    if isinstance(text, str):  # Check if the text is a string
        return analyzer.polarity_scores(text)['compound']
    
    return 0  # Return 0 for non-string or empty content

reddit_df['submission_sentiment'] = reddit_df['submission'].apply(get_sentiment_score)
reddit_df['title_sentiment'] = reddit_df['title'].apply(get_sentiment_score)


sentiment_df = reddit_df[['date', 'submission_sentiment', 'title_sentiment']].copy()

sentiment_df = sentiment_df.groupby('date').mean().reset_index()

sentiment_df['sentiment'] = sentiment_df[['submission_sentiment', 'title_sentiment']].mean(axis=1)
sentiment_df = sentiment_df[['date', 'sentiment']]
sentiment_df.to_csv('Data/sentiment_data.csv', index=False)'''

bitcoin_df = pd.read_csv('Data/BitcoinPriceData.csv', delimiter=';')
bitcoin_df['timeOpen'] = pd.to_datetime(bitcoin_df['timeOpen'])
sentiment_df = pd.read_csv('Data/sentiment_data.csv')
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])


bitcoin_df['timeOpen'] = bitcoin_df['timeOpen'].dt.tz_localize(None)


merged_df = pd.merge(sentiment_df, bitcoin_df, left_on='date', right_on='timeOpen', how='inner')

#Display the merged DataFrame
print(merged_df.head())
print(bitcoin_df.head())
print(sentiment_df.head())
# Plot sentiment over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='sentiment', data=sentiment_df, marker='o')
plt.title('Reddit Submission Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.show()

# Plot sentiment vs. Bitcoin closing price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sentiment', y='close', data=merged_df)
plt.title('Sentiment vs. Bitcoin Closing Price')
plt.xlabel('Sentiment Score')
plt.ylabel('Bitcoin Closing Price')
plt.grid(True)
plt.show()

# Calculate correlation
correlation = merged_df['sentiment'].corr(merged_df['close'])
print(f"Correlation between sentiment and Bitcoin closing price: {correlation:.2f}")

# 3. New: Dual-axis plot of Sentiment and Bitcoin Price over time
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Date')
ax1.set_ylabel('Sentiment Score', color='tab:blue')
ax1.plot(merged_df['date'], merged_df['sentiment'], color='tab:blue', marker='o', label='Sentiment')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Bitcoin Closing Price', color='tab:orange')
ax2.plot(merged_df['date'], merged_df['close'], color='tab:orange', marker='o', label='Bitcoin Price')
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.title('Sentiment and Bitcoin Price Over Time')
fig.tight_layout()
plt.grid(True)
plt.show()

# 4. New: Rolling average (7-day) of Sentiment and Price
merged_df['sentiment_roll'] = merged_df['sentiment'].rolling(window=7, min_periods=1).mean()
merged_df['close_roll'] = merged_df['close'].rolling(window=7, min_periods=1).mean()
plt.figure(figsize=(12, 6))
plt.plot(merged_df['date'], merged_df['sentiment_roll'], label='7-Day Rolling Sentiment', color='blue')
plt.plot(merged_df['date'], merged_df['close_roll'] / merged_df['close_roll'].max(), label='7-Day Rolling Price (Normalized)', color='orange')
plt.title('7-Day Rolling Average of Sentiment and Normalized Bitcoin Price')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# 5. New: Histogram of Sentiment Scores
plt.figure(figsize=(10, 6))
sns.histplot(merged_df['sentiment'], bins=20, kde=True)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 6. New: Cross-correlation between Sentiment and Price with lags
lags = range(-10, 11)  # Check lags from -10 to +10 days
cross_corr = [merged_df['sentiment'].corr(merged_df['close'].shift(lag)) for lag in lags]
plt.figure(figsize=(10, 6))
plt.stem(lags, cross_corr)
plt.title('Cross-Correlation of Sentiment and Bitcoin Price Across Lags')
plt.xlabel('Lag (Days)')
plt.ylabel('Correlation')
plt.grid(True)
plt.show()

