import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import ccf

volatility_df = pd.read_csv('Data/bitcoin_volatility.csv')


sentiment_df = pd.read_csv('Data/sentiment_data.csv')


print(volatility_df.columns)
print(sentiment_df.columns)
volatility_df['date'] = pd.to_datetime(volatility_df['date'])
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

merged_df = pd.merge(
    volatility_df,
    sentiment_df,
    on='date',
    how='outer'  
)

merged_df = merged_df.sort_values('date').reset_index(drop=True)
cleaned_df = merged_df.dropna()
print(cleaned_df.head())


pearson_corr = cleaned_df['submission_sentiment'].corr(cleaned_df['parkinson_vol'])
print(f"Pearson Correlation: {pearson_corr:.3f}")

spearman_corr = cleaned_df['submission_sentiment'].corr(cleaned_df['parkinson_vol'], method='spearman')
print(f"Spearman Correlation: {spearman_corr:.3f}")


granger_test = grangercausalitytests(cleaned_df[['parkinson_vol', 'submission_sentiment']], maxlag=7)
granger_test_reverse = grangercausalitytests(
    cleaned_df[['submission_sentiment', 'parkinson_vol']], maxlag=7
)


# Drop rows with missing sentiment (optional but good practice)

# Get sentiment range
min_sentiment = sentiment_df['submission_sentiment'].min()
max_sentiment = sentiment_df['submission_sentiment'].max()
print(min_sentiment, max_sentiment)

print("Summary Statistics for Sentiment:")
print(sentiment_df['submission_sentiment'].describe())

# Check for skewness
skewness = sentiment_df['submission_sentiment'].skew()
print(f"\nSkewness: {skewness:.3f}")

# Classify sentiment buckets
sentiment_df['sentiment_class'] = pd.cut(sentiment_df['submission_sentiment'], 
                                bins=[-1, -0.05, 0.05, 1], 
                                labels=['Negative', 'Neutral', 'Positive'])





'''
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Date')
ax1.set_ylabel('Sentiment Score', color='tab:blue')
ax1.plot(merged_df['date'], merged_df['submission_sentiment'], color='tab:blue', marker='o', label='Sentiment')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Bitcoin Volatility', color='tab:orange')
ax2.plot(merged_df['date'], merged_df['parkinson_vol'], color='tab:orange', marker='o', label='Bitcoin Price')
ax2.tick_params(axis='y', labelcolor='tab:orange')
plt.title('Sentiment and Bitcoin Price Over Time')
fig.tight_layout()
plt.grid(True)
plt.show()

cleaned_df['rolling_corr'] = cleaned_df['submission_sentiment'].rolling(window=30).corr(cleaned_df['parkinson_vol'])
plt.figure(figsize=(12, 6))
plt.plot(cleaned_df['date'], cleaned_df['rolling_corr'], color='purple')
plt.title("30-Day Rolling Correlation: Sentiment vs. Volatility")
plt.xlabel("Date")
plt.ylabel("Correlation Coefficient")
plt.axhline(0, linestyle='--', color='gray')  # Zero correlation line
plt.grid()
plt.show() 


plt.figure(figsize=(12, 6))
plt.plot(cleaned_df['date'], cleaned_df['submission_sentiment'], label='Sentiment', color='blue', alpha=0.7)
plt.plot(cleaned_df['date'], cleaned_df['parkinson_vol'], label='Volatility', color='red', alpha=0.7)
plt.title("Bitcoin Reddit Sentiment vs. Market Volatility (Over Time)")
plt.xlabel("Date")
plt.ylabel("Value (Normalized)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='submission_sentiment', y='parkinson_vol', data=cleaned_df, alpha=0.6)
plt.title("Bitcoin Reddit Sentiment vs. Volatility (Scatter Plot)")
plt.xlabel("Sentiment Score")
plt.ylabel("Parkinson Volatility")
plt.show()

cross_corr = ccf(cleaned_df['submission_sentiment'], cleaned_df['parkinson_vol'], adjusted=False)
plt.stem(range(len(cross_corr)), cross_corr)
plt.title("Lagged Cross-Correlation: Sentiment → Volatility")
plt.xlabel("Lag (Days)")
plt.ylabel("Correlation")
plt.show()
'''