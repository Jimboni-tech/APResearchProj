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

median_sentiment = cleaned_df['submission_sentiment'].median()
median_volatility = cleaned_df['parkinson_vol'].median()

print(f"Median Sentiment: {median_sentiment:.3f}")
print(f"Median Volatility: {median_volatility:.3f}")



fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()

ax1.plot(cleaned_df['date'], cleaned_df['parkinson_vol'], 'b-', label='Volatility')
ax2.plot(cleaned_df['date'], cleaned_df['submission_sentiment'], 'r-', label='Sentiment')

ax1.set_xlabel('Date')
ax1.set_ylabel('Volatility', color='b')
ax2.set_ylabel('Sentiment', color='r')
plt.title("Sentiment and Volatility Over Time")
plt.legend(loc='upper left')
plt.show()