import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from ml_model_utils import train_ml_model
from textblob import TextBlob

def rule_based_analysis(text):
    # rule-based sentiment analysis using TextBlob library
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity < 0 and subjectivity > 0.5:
        return 1  # toxic
    else: 
        return 0  # non-toxic
    
def ml_based_analysis(texts, model, vectorizer):
    # ml-based sentiment analysis (vectorized for multiple texts)
    X_vec = vectorizer.transform(texts)  # multiple texts to vectors
    probs = model.predict_proba(X_vec)[:, 1]  # probability of toxic class
    preds = (probs >= 0.5).astype(int)  # convert to 0 or 1
    return preds

def monte_carlo_simulation(df, n_runs=100):
    results = []
    all_preds = []
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        
        # random split for train and test set
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_tweet'], df['Toxicity'], test_size=0.2, random_state=None  # random split each time
        )

        # train ml model using imported function from ml utils
        model, vectorizer = train_ml_model(X_train, y_train)

        ml_preds = ml_based_analysis(X_test, model, vectorizer)

        rule_preds = X_test.apply(rule_based_analysis)

        hybrid_preds = ((0.5 * rule_preds) + (0.5 * ml_preds) >= 0.5).astype(int)

        # metrics
        result = {
            'run': run + 1,
            'ml_accuracy': accuracy_score(y_test, ml_preds),
            'ml_f1': f1_score(y_test, ml_preds),
            'rule_accuracy': accuracy_score(y_test, rule_preds),
            'rule_f1': f1_score(y_test, rule_preds),
            'hybrid_accuracy': accuracy_score(y_test, hybrid_preds),
            'hybrid_f1': f1_score(y_test, hybrid_preds)
        }
        results.append(result)
        
        all_preds.append({
            'run': run + 1,
            'y_test': np.array(y_test),
            'ml_preds': ml_preds,
            'rule_preds': np.array(rule_preds),
            'hybrid_preds': hybrid_preds
        })

    return pd.DataFrame(results), all_preds

        
        
# load dataset
df = pd.read_csv('cleaned_tweets_df.csv')

start_time = time.perf_counter() # this part times how long the serial version runs (for 100 repeated mc simulations)
# apply monte carlo runs to the tweets dataframe
results_df, all_preds = monte_carlo_simulation(df, n_runs=100)
end_time = time.perf_counter()

# ----------------visualization of data---------------------------------
# barplot for mean scores
mean_scores = results_df[['ml_accuracy', 'rule_accuracy', 'hybrid_accuracy',
                          'ml_f1', 'rule_f1', 'hybrid_f1']].mean()

mean_scores.plot(kind='bar', figsize=(10,6), title='Average Performance (Serial Version)')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# boxplot for distribution of accuracy across 100 runs
sns.boxplot(data=results_df[['ml_accuracy', 'rule_accuracy', 'hybrid_accuracy']])
plt.title('Accuracy Distribution (Serial Version)')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# boxplot for F1 score distribution
sns.boxplot(data=results_df[['ml_f1', 'rule_f1', 'hybrid_f1']])
plt.title('F1 Score Distribution (Serial Version)')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.show()

# Confusion Matrices (best run)
# Find best hybrid run
best_run_idx = results_df['hybrid_accuracy'].idxmax()
best_preds = all_preds[best_run_idx]

y_test = best_preds['y_test']
ml_preds = best_preds['ml_preds']
rule_preds = best_preds['rule_preds']
hybrid_preds = best_preds['hybrid_preds']

# ml-based confusion matrix
cm_ml = confusion_matrix(y_test, ml_preds)
disp_ml = ConfusionMatrixDisplay(confusion_matrix=cm_ml, display_labels=['Non-Toxic', 'Toxic'])
disp_ml.plot(cmap='Blues')
plt.title('ML Model Confusion Matrix (Best Run)')
plt.show()

# rule-based confusion matrix
cm_rule = confusion_matrix(y_test, rule_preds)
disp_rule = ConfusionMatrixDisplay(confusion_matrix=cm_rule, display_labels=['Non-Toxic', 'Toxic'])
disp_rule.plot(cmap='Purples')
plt.title('Rule-based Confusion Matrix (Best Run)')
plt.show()

# hybrid confusion matrix
cm_hybrid = confusion_matrix(y_test, hybrid_preds)
disp_hybrid = ConfusionMatrixDisplay(confusion_matrix=cm_hybrid, display_labels=['Non-Toxic', 'Toxic'])
disp_hybrid.plot(cmap='Greens')
plt.title('Hybrid Confusion Matrix (Best Run)')
plt.show()

# create csv file from the generated dataframe
results_df.to_csv('sentiment_analysis_serial.csv', index=False)

print(results_df.describe())  # show mean, std, etc.
print(f"Serial run time: {end_time - start_time:.2f} seconds")
