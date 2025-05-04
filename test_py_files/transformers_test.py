import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load your dataset (adjust the path)
df = pd.read_csv('cleaned_tweets_df.csv')

# Assume 'tweet' is the text column and 'label' is 0/1 for toxic or not
X = df['cleaned_tweet']
y = df['Toxicity']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression (can swap to MultinomialNB for Naive Bayes)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))


