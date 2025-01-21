import praw
from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')
plt.style.use('ggplot')
reddit = praw.Reddit(
    client_id="_pKCPsBoDa1S3pYDVmeGqQ",
    client_secret="toDp5RAErDA6KL_7zzUJiu4KT-wjuA",
    password="1playclarinetF",
    user_agent="Cloud-General",
    username="Cloud-General",
)
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
data = set()
sia = SIA()
rawData = []
keywordList = '"Bitcoin", "$BTC", "BTC"'
subreddit = reddit.subreddit('crypto+CryptoCurrency+CryptoMarkets+CryptoMoonShots+Crypto_com+Bitcoin+CryptoTechnology+CryptoMars+btc+CryptoCurrencyTrading+CryptoMoon+Crypto_Currency_News+Crypto_General').search(keywordList, limit=None, time_filter='day')
subreddit2 = reddit.subreddit('all').search(keywordList, limit=None, time_filter='day')
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
display.display(df)
print(len(df))

df.to_csv('10-30data.csv', header = False, encoding = "utf-8", index = False)
df2 = df[['Headline', 'compound']]
df2.to_csv('test.csv', header = False, encoding = "utf-8", index = False)

