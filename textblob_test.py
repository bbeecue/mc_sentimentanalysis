import pandas as pd
from textblob import TextBlob


tweets_df = pd.read_csv("FinalBalancedDataset.csv")

#print(tweets_df.head(10))

tweets_df.dropna(inplace=True)
tweets_df = tweets_df[['Toxicity', 'tweet']] 

def textblob_analysis(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    return 'Toxic' if polarity < -0.1 else 'Non-toxic'


# note: polarity meseaures the sentiment of the text
#0.0 neutral
#1.0 very positive polarity (sentiment)
#-1.0 very negative polarity (sentiment)

print(textblob_analysis("I am beautiful!"))