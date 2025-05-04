from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def train_ml_model(X, y):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    return model, vectorizer

def save_model(model, vectorizer, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

def load_model(model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
