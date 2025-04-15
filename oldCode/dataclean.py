import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('Data/BitcoinSubmissions.csv')

def clean_text(text):
  
    text = str(text)

    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)

    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = text.strip()

    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
    
    return ' '.join(filtered_words)

df['cleaned_submission'] = df['submission'].apply(clean_text)
clean_df = df[
    (df['cleaned_submission'] != "") &
    (~df['cleaned_submission'].isin(['removed', 'nan', 'none', 'null', 'deleted'])) &
    (df['cleaned_submission'].notna())
]

print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {clean_df.shape}")
print(f"Removed {len(df) - len(clean_df)} empty submissions")

assert all(col in clean_df.columns for col in df.columns)
assert 'cleaned_submission' in clean_df.columns

