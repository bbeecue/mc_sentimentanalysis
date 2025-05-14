import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import string
from tqdm import tqdm
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from ml_model_utils import train_ml_model
from textblob import TextBlob
from collections import Counter

def add_random_noise(text, noise_level=0.05):
    words = text.split()
    n_inserts = max(1, int(len(words) * noise_level))
    for _ in range(n_inserts):
        rand_word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3,7)))
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, rand_word)
    return ' '.join(words)

def random_char_swap(text, alteration_level=0.02):
    chars = list(text)
    n_swaps = max(1, int(len(chars) * alteration_level))
    for _ in range(n_swaps):
        idx1, idx2 = random.sample(range(len(chars)), 2)
        chars[idx1], chars[idx2] = chars[idx2], chars[idx1]
    return ''.join(chars)

def apply_random_alterations(text):
    text = add_random_noise(text)
    text = random_char_swap(text)
    return text

def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

def rule_based_score(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity  # between -1 to 1

def ml_based_score(text, model, vectorizer):
    X_vec = vectorizer.transform([text])
    probas = model.predict_proba(X_vec)[0]  # [neg, neu, pos]
    polarity_score = probas[2] - probas[0]  # pos - neg
    return polarity_score

def hybrid_sentiment_score(text, model, vectorizer):
    rule_score = rule_based_score(text)
    ml_score = ml_based_score(text, model, vectorizer)
    ml_score_sigmoid = bipolar_sigmoid(ml_score * 5)  # smooth the ML score
    hybrid_score = 0.5 * rule_score + 0.5 * ml_score_sigmoid  # equal weight

    # convert hybrid score to class label
    if hybrid_score > 0.1:
        label = 'positive'
    elif hybrid_score < -0.1:
        label = 'negative'
    else:
        label = 'neutral'
    
    return label

def rule_based_analysis(text):
    # rule-based sentiment analysis using TextBlob library
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
   #subjectivity = blob.sentiment.subjectivity
    if polarity < 0:
        return "negative"  
    elif polarity == 0: 
        return "neutral" 
    else:
        return "positive"
    
def ml_based_analysis(texts, model, vectorizer):
    # ml-based sentiment analysis (vectorized for multiple texts)
    X_vec = vectorizer.transform(texts)  # multiple texts to vectors
    predictions = model.predict(X_vec)
    return predictions  

def hybrid_based_analysis (rule_preds, ml_preds):
    hybrid_preds = []
    for rule_pred, ml_pred in zip(rule_preds, ml_preds):
        rule_pred = str(rule_pred)
        ml_pred = str(ml_pred)
        votes = [rule_pred, ml_pred]
        common = Counter(votes).most_common(1)[0][0]
        hybrid_preds.append(common)
    return np.array(hybrid_preds)

def monte_carlo_simulation(df, n_runs):
    results = []
    all_preds = []
    
    sampled_df = df.sample(n=10000, random_state=None)
    
    for run in tqdm(range(n_runs), desc="Monte Carlo Simulation"):
        # random split for train and test set
        X_train, X_test, y_train, y_test = train_test_split(
            sampled_df['text'], sampled_df['sentiment'], test_size=0.2, random_state=None  # random split each time
        )
        
        X_train_noisy = X_train.apply(apply_random_alterations)
        X_test_noisy = X_test.apply(apply_random_alterations)

        # train ml model using imported function from ml utils
        model, vectorizer = train_ml_model(X_train_noisy, y_train)

        ml_preds = ml_based_analysis(X_test_noisy, model, vectorizer)

        rule_preds = X_test_noisy.apply(rule_based_analysis)

        hybrid_preds = X_test_noisy.apply(lambda x: hybrid_sentiment_score(x, model, vectorizer))

        # metrics
        result = {
            'run': run + 1,
            'ml_accuracy': accuracy_score(y_test, ml_preds),
            'ml_f1': f1_score(y_test, ml_preds, average='macro'),
            'rule_accuracy': accuracy_score(y_test, rule_preds),
            'rule_f1': f1_score(y_test, rule_preds, average='macro'),
            'hybrid_accuracy': accuracy_score(y_test, hybrid_preds),
            'hybrid_f1': f1_score(y_test, hybrid_preds, average='macro')
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
df = pd.read_csv('preprocessed_tweets_df.csv')

start_time = time.perf_counter() # this part times how long the serial version runs (for 100 repeated mc simulations)
# apply monte carlo runs to the tweets dataframe
results_df, all_preds = monte_carlo_simulation(df, n_runs=500)
end_time = time.perf_counter()

# ----------------visualization of data---------------------------------
# barplot for mean scores
mean_scores = results_df[['ml_accuracy', 'rule_accuracy', 'hybrid_accuracy',
                          'ml_f1', 'rule_f1', 'hybrid_f1']].mean()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#1f77b4','#ff7f0e','#2ca02c']  

mean_scores.plot(color=colors, kind='bar', figsize=(10,6), title='Average Performance (Serial Version)', )
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

labels = ['negative', 'neutral', 'positive']

# ml-based confusion matrix
cm_ml = confusion_matrix(y_test, ml_preds, labels=labels)
disp_ml = ConfusionMatrixDisplay(confusion_matrix=cm_ml, display_labels=['Negative', 'Neutral',  'Positive'])
disp_ml.plot(cmap='Blues')
plt.title('ML Model Confusion Matrix (Best Run)')
plt.show()

# rule-based confusion matrix
cm_rule = confusion_matrix(y_test, rule_preds, labels=labels)
disp_rule = ConfusionMatrixDisplay(confusion_matrix=cm_rule, display_labels=['Negative', 'Neutral',  'Positive'])
disp_rule.plot(cmap='Purples')
plt.title('Rule-based Confusion Matrix (Best Run)')
plt.show()

# hybrid confusion matrix
cm_hybrid = confusion_matrix(y_test, hybrid_preds, labels=labels)
disp_hybrid = ConfusionMatrixDisplay(confusion_matrix=cm_hybrid, display_labels=['Negative', 'Neutral',  'Positive'])
disp_hybrid.plot(cmap='Greens')
plt.title('Hybrid Confusion Matrix (Best Run)')
plt.show()



# create csv file from the generated dataframe
results_df.to_csv('sentiment_analysis_serial.csv', index=False)

print(results_df.describe())  # show mean, std, etc.
print(f"Serial run time: {end_time - start_time:.2f} seconds")
