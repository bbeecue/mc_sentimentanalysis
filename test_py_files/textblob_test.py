import pandas as pd
import re
from textblob import TextBlob


tweets_analysis_df = pd.read_csv('cleaned_tweets_df.csv')
#print(tweets_df.head(10))


def textblob_analysis(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity < 0 and subjectivity > 0.5:
        return '1'
    else: 
        return '0'


# note: polarity meseaures the sentiment of the text
#0.0 neutral
#1.0 very positive polarity (sentiment)
#-1.0 very negative polarity (sentiment)

# print(tweets_df.tail(10))

tweets_analysis_df['textblob_pred'] = tweets_analysis_df['cleaned_tweet'].apply(textblob_analysis)
print(tweets_analysis_df)