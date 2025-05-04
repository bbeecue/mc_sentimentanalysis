import pandas as pd
from textblob import TextBlob
from ml_model_utils import train_ml_model

def rule_based_analysis(text):
    # rule-based sentiment analysis using TextBlob library
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity < 0 and subjectivity > 0.5:
        return 1  # toxic
    else: 
        return 0  # non-toxic
    
def ml_based_analysis(text, model, vectorizer):
    # ml-bassed sentiment analysis using scikit + logistic regression
    X_vec = vectorizer.transform([text])  # single text to vector
    prob = model.predict_proba(X_vec)[0, 1]  # probability of toxic class
    return int(prob >= 0.5)  # convert to 0 or 1

# load dataset
df = pd.read_csv('cleaned_tweets_df.csv')
# train ml model
X = df['cleaned_tweet']
y = df['Toxicity']
model, vectorizer = train_ml_model(X, y)

# application of different approaches
df['rule_based_pred'] = df['cleaned_tweet'].apply(rule_based_analysis)

df['ml_based_pred'] = df['cleaned_tweet'].apply(lambda x: ml_based_analysis(x, model, vectorizer))

df['hybrid_based_pred'] = ((0.5 * df['rule_based_pred']) + (0.5 * df['ml_based_pred']) >= 0.5).astype(int)

# create csv file from the generated dataframe
df.to_csv('tweets_with_predictions.csv', index=False)

print(df.head())
