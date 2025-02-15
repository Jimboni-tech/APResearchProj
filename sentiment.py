import praw
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import matplotlib.pyplot as plt
import seaborn as sns

reddit = praw.Reddit(

)

sia = SIA()

subreddit_string = (
    'crypto+CryptoCurrency+CryptoMarkets+CryptoMoonShots+Crypto_com+Bitcoin+CryptoTechnology+'
    'CryptoMars+btc+CryptoCurrencyTrading+CryptoMoon+Crypto_Currency_News+Crypto_General+ethereum+'
    'blockchain+Cryptocurrency+cryptonews+Cryptos+DeFi+Litecoin+dogecoin+XRP+Solana+Altcoin+NFTs+cryptotokens'
    '+cryptoinvesting+cryptotrading+blockchainnews+cryptocurrencynews+cryptomarket'
)


def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return tokens


def label_sentiment(row):
    if row['compound'] >= 0.05:
        return 'Positive'
    elif row['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def clean_and_sort_data(df):
    df = df[df['Upvotes'] > 0.2]  
    df = df[df['Text'].str.len() > 50]
    df = df[abs(df['compound']) < 0.9] 
    df = df.drop_duplicates(subset='Headline', keep='first') 
    df = df.sort_values(by='Upvotes', ascending=False)  
    df.reset_index(drop=True, inplace=True)
    return df


rawData = []
keywordList = '"Bitcoin", "$BTC", "BTC"'
subreddit = reddit.subreddit(subreddit_string).search(keywordList, limit=None, time_filter='day')

for submission in subreddit:
    if not submission.stickied:
        if len(submission.selftext) > 0:
            pol_score = sia.polarity_scores(submission.selftext)
            pol_score['Headline'] = submission.title
            pol_score['Text'] = submission.selftext
            pol_score['Date Created'] = submission.created_utc
            pol_score['Upvotes'] = submission.upvote_ratio
            rawData.append(pol_score)


df = pd.DataFrame.from_records(rawData)


df_cleaned = clean_and_sort_data(df)


df_cleaned['Sentiment Label'] = df_cleaned.apply(label_sentiment, axis=1)

print(df_cleaned.head())

plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment Label', data=df_cleaned, palette='Set2')
plt.title('Sentiment Distribution of Posts')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


df_cleaned.to_csv('wkOfJan27Data.csv', header=True, encoding="utf-8", index=False)