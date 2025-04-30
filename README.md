# 📊 Reddit Sentiment & Bitcoin Volatility Analysis

This project explores the relationship between public sentiment on the r/Bitcoin subreddit and Bitcoin market volatility using Python, VADER sentiment analysis, and time-series statistical methods. It was originally created as part of a high school AP Research project.

---

## 🧠 Overview

- **Abstract:**  
  This study investigates the relationship between public sentiment on the subreddit r/Bitcoin and Bitcoin market volatility. Utilizing sentiment analysis with VADER and calculating volatility using Parkinson’s formula, the study found a weak negative correlation between sentiment and volatility. Granger causality tests revealed that while Bitcoin volatility significantly influences Reddit sentiment, sentiment does not have predictive power over volatility.

- **Keywords:**  
  *Bitcoin, Cryptocurrency, Reddit, Sentiment Analysis, Market Volatility, Granger Causality, VADER, Social Media Analytics, Time Series Analysis*

---

## 📊 Visualizations

### 📉 Regression Line
![Regression Line](https://github.com/user-attachments/assets/32aff9a5-bee2-4e98-ae65-cc604bdc8825)

### 📈 Sentiment Distribution
![Sentiment Distribution](https://github.com/user-attachments/assets/2af99ae9-6480-4875-9841-a04857a99f35)

### 📆 Time Series Dual Analysis
![Time Series Dual Analysis](https://github.com/user-attachments/assets/89de8628-7483-4795-b193-81e1467f68cb)

---

## 🔍 Implications

- Granger causality analysis indicates that **volatility Granger-causes sentiment**, but not vice versa.
- This suggests that **Reddit sentiment is more reactive** than predictive.
- Sentiment may act as a **real-time stress indicator**, rather than a forecasting tool for price movement.
- Investors should not rely on sentiment as a trading signal, as correlation does **not imply causation**.

---

## ⚠️ Limitations

- **VADER Accuracy:** While useful, VADER struggles with **sarcasm, slang, and nuance** in social media.
- **Spam & Noise:** Reddit posts may include irrelevant content (memes, bots), impacting sentiment averages.
- **External Influences:** Political or economic events may drive volatility independently of Reddit sentiment.
- **Hardware Constraints:** Data was sampled due to hardware limitations (MacBook Air with M3 chip), reducing overall scope.
- **Storage Size:** Full Reddit datasets (via Pushshift) were **terabyte-scale**, limiting full exploration.

---

## 🚀 Future Research Directions

- Train **LLMs or regression models** on combined sentiment and volatility datasets.
- Use transformer-based models (e.g., **BERT, RoBERTa, FinBERT**) fine-tuned on crypto-specific data.
- Analyze **topic-specific sentiment** (e.g., technical vs. investing) for deeper correlation patterns.
- Explore **multi-platform sentiment** (Reddit, Twitter, Telegram) for broader market insight.

---
