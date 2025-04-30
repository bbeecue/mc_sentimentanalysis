import pandas as pd
import re


tweets_df = pd.read_csv('FinalBalancedDataset.csv', encoding= 'utf8', engine = 'python')
tweets_df.dropna(inplace=True)
tweets_df = tweets_df[['Toxicity', 'tweet']] 

def fix_encoding(text):
    try:
        return text.encode('latin1').decode('utf-8')
    except:
        return text 
    
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["                      # emoji ranges
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\u2600-\u26FF"          # miscellaneous symbols
        u"\u2700-\u27BF"          # dingbats
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()  # normalize spaces
    return text

tweets_df['cleaned_tweet'] = tweets_df['tweet'].apply(fix_encoding).apply(remove_emojis).apply(clean_text)
cleaned_tweets_df = tweets_df[['Toxicity', 'cleaned_tweet']]

print(cleaned_tweets_df)

cleaned_tweets_df.to_csv('cleaned_tweets_df.csv', index=False)