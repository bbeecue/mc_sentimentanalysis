import pandas as pd

tweets_df = pd.read_csv('Tweets.csv', encoding= 'utf8', engine = 'python')
tweets_df.dropna(inplace=True)
tweets_df = tweets_df[['text', 'sentiment']] 


print(tweets_df)

tweets_df.to_csv('preprocessed_tweets_df.csv', index=False)