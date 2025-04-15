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

sns.lmplot(x='submission_sentiment', y='parkinson_vol', data=cleaned_df, 
           line_kws={'color': 'red'}, scatter_kws={'alpha': 0.5})
plt.title("Sentiment vs. Volatility (with Regression Line)")
plt.xlabel("Sentiment Score")
plt.ylabel("Parkinson Volatility")
plt.show()