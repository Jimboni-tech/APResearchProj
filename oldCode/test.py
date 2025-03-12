import os
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import matplotlib.pyplot as plt
import kagglehub

def labelSentiment(row):
    if row['compound'] >= 0.05:
        return 'Positive'
    elif row['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Download latest version
path = kagglehub.dataset_download("leukipp/reddit-crypto-data")

files = []
for folder, _, dataset in os.walk('/Users/jimmyzhou/.cache/kagglehub/datasets/leukipp/reddit-crypto-data/versions/325'):
    files = files + [os.path.join(folder, x) for x in dataset if x.endswith('.csv')]
dfs = {os.path.basename(os.path.dirname(x)): pd.read_csv(x) for x in files}
df = dfs['cryptocurrency']

df['created'] = pd.to_datetime(df['created'], unit='s')
df['date'] = df['created'].dt.date

dataset = df[['created', 'selftext', 'date']]  # Include 'date' in the dataset
dataset = dataset.dropna(subset=['selftext'])  # Remove NaN values
dataset = dataset[dataset['selftext'].str.strip() != '']  # Remove empty strings
dataset = dataset[~dataset['selftext'].str.lower().isin(["[removed]", "[deleted]"])]

sia = SIA()
sentimentResults = []

for text in dataset['selftext']:
     pol_score = sia.polarity_scores(str(text))
     pol_score['Sentiment Label'] = labelSentiment(pol_score)
     sentimentResults.append(pol_score)

sentiment_df = pd.DataFrame(sentimentResults)
final_dataset = pd.concat([dataset.reset_index(drop=True), sentiment_df], axis=1)

# Ensure 'date' column exists in final_dataset
if 'date' not in final_dataset.columns:
    raise KeyError("Column 'date' is missing in final_dataset.")

final_dataset = final_dataset.dropna(subset=['compound']) 
final_dataset = final_dataset[final_dataset['compound'] >= -1.0] 
final_dataset = final_dataset[final_dataset['compound'] <= 1.0]  


daily_sentiment = final_dataset.groupby('date')['compound'].mean().reset_index()
daily_sentiment.rename(columns={'compound': 'average_sentiment'}, inplace=True)

bitcoin_df = pd.read_csv('bitcoin_data.csv', delimiter = ";")
bitcoin_df = bitcoin_df[['timeOpen', 'volume']]
bitcoin_df['timeOpen'] = pd.to_datetime(bitcoin_df['timeOpen']).dt.date
bitcoin_df.rename(columns={'timeOpen': 'date'}, inplace=True)



correlationdf = pd.merge(bitcoin_df, daily_sentiment, on = 'date')
print(correlationdf)

'''plt.figure(figsize=(12, 6))
plt.plot(daily_sentiment['date'], daily_sentiment['average_sentiment'], marker='o', linestyle='-')
plt.title('Average Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()'''

