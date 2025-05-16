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
from collections import Counter
import multiprocessing as mp
import concurrent.futures

def add_noise(text, noise_level=0.05):
    words = text.split()
    n_inserts = max(1, int(len(words) * noise_level))
    for _ in range(n_inserts):
        rand_word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3,7)))
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, rand_word)
    return ' '.join(words)

def char_swap(text, alteration_level=0.02):
    chars = list(text)
    n_swaps = max(1, int(len(chars) * alteration_level))
    for _ in range(n_swaps):
        idx1, idx2 = random.sample(range(len(chars)), 2)
        chars[idx1], chars[idx2] = chars[idx2], chars[idx1]
    return ''.join(chars)

def apply_alterations(text):
    text = add_noise(text)
    text = char_swap(text)
    return text

def apply_random(series, fraction=0.4):
    noisy_series = series.copy()
    noisy_indices = np.random.choice(series.index, size=int(len(series)*fraction), replace=False)
    noisy_series.loc[noisy_indices] = series.loc[noisy_indices].apply(apply_alterations)
    return noisy_series

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
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity < 0:
        return "negative"
    elif polarity == 0:
        return "neutral"
    else:
        return "positive"

def ml_based_analysis(texts, model, vectorizer):
    X_vec = vectorizer.transform(texts)
    predictions = model.predict(X_vec)
    return predictions

def hybrid_based_analysis(rule_preds, ml_preds):
    hybrid_preds = []
    for rule_pred, ml_pred in zip(rule_preds, ml_preds):
        rule_pred = str(rule_pred)
        ml_pred = str(ml_pred)
        votes = [rule_pred, ml_pred]
        common = Counter(votes).most_common(1)[0][0]
        hybrid_preds.append(common)
    return np.array(hybrid_preds)


def run_single_simulation(run_idx, df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['sentiment'], test_size=0.2, random_state=None
    )

    X_train_noisy = apply_random(X_train, fraction=0.4)
    X_test_noisy = apply_random(X_test, fraction=0.4)
    
    model, vectorizer = train_ml_model(X_train_noisy, y_train)
    
    ml_preds = ml_based_analysis(X_test_noisy, model, vectorizer)
    rule_preds = X_test_noisy.apply(rule_based_analysis)
    hybrid_preds = X_test_noisy.apply(lambda x: hybrid_sentiment_score(x, model, vectorizer))

    result = {
        'run': run_idx + 1,
        'ml_accuracy': accuracy_score(y_test, ml_preds),
        'ml_f1': f1_score(y_test, ml_preds, average='macro'),
        'rule_accuracy': accuracy_score(y_test, rule_preds),
        'rule_f1': f1_score(y_test, rule_preds, average='macro'),
        'hybrid_accuracy': accuracy_score(y_test, hybrid_preds),
        'hybrid_f1': f1_score(y_test, hybrid_preds, average='macro')
    }

    preds = {
        'run': run_idx + 1,
        'y_test': np.array(y_test),
        'ml_preds': ml_preds,
        'rule_preds': np.array(rule_preds),
        'hybrid_preds': hybrid_preds
    }

    return result, preds

def run_single_simulation_wrapper(args):
    return run_single_simulation(*args)

def monte_carlo_simulation_parallel(df, n_runs):
    results = []
    all_preds = []

    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing...")
    
    sampled_df = df.sample(n=10000, random_state=None)

    tasks = [(i, sampled_df) for i in range(n_runs)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(run_single_simulation_wrapper, task) for task in tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=n_runs, desc="Monte Carlo Simulations"):
            result, preds = future.result()
            results.append(result)
            all_preds.append(preds)

    return pd.DataFrame(results), all_preds


if __name__ == "__main__":
    df = pd.read_csv('preprocessed_tweets_df.csv')

    start_time = time.perf_counter()
    results_df, all_preds = monte_carlo_simulation_parallel(df, n_runs=500)
    end_time = time.perf_counter()

    # save to csv
    results_df.to_csv('sentiment_analysis_parallel.csv', index=False)

    print(results_df.describe())
    print(f"Parallel run time: {end_time - start_time:.2f} seconds")

    # ----------------visualization of data---------------------------------
    # barplot for mean scores
    mean_scores = results_df[['ml_accuracy', 'rule_accuracy', 'hybrid_accuracy',
                              'ml_f1', 'rule_f1', 'hybrid_f1']].mean()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#1f77b4','#ff7f0e','#2ca02c']  

    mean_scores.plot(color=colors, kind='bar', figsize=(10,6), title='Average Performance (Parallel Version)')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    sns.boxplot(data=results_df[['ml_accuracy', 'rule_accuracy', 'hybrid_accuracy']])
    plt.title('Accuracy Distribution (Parallel Version)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.show()

    sns.boxplot(data=results_df[['ml_f1', 'rule_f1', 'hybrid_f1']])
    plt.title('F1 Score Distribution (Parallel Version)')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.show()

    # confusion Matrices (best run)
    best_run_idx = results_df['hybrid_accuracy'].idxmax()
    best_preds = all_preds[best_run_idx]

    y_test = best_preds['y_test']
    ml_preds = best_preds['ml_preds']
    rule_preds = best_preds['rule_preds']
    hybrid_preds = best_preds['hybrid_preds']

    labels = ['negative', 'neutral', 'positive']

    cm_ml = confusion_matrix(y_test, ml_preds, labels=labels)
    disp_ml = ConfusionMatrixDisplay(confusion_matrix=cm_ml, display_labels=['Negative', 'Neutral',  'Positive'])
    disp_ml.plot(cmap='Blues')
    plt.title('ML Model Confusion Matrix (Best Run)')
    plt.show()

    cm_rule = confusion_matrix(y_test, rule_preds, labels=labels)
    disp_rule = ConfusionMatrixDisplay(confusion_matrix=cm_rule, display_labels=['Negative', 'Neutral',  'Positive'])
    disp_rule.plot(cmap='Purples')
    plt.title('Rule-based Confusion Matrix (Best Run)')
    plt.show()

    cm_hybrid = confusion_matrix(y_test, hybrid_preds, labels=labels)
    disp_hybrid = ConfusionMatrixDisplay(confusion_matrix=cm_hybrid, display_labels=['Negative', 'Neutral',  'Positive'])
    disp_hybrid.plot(cmap='Greens')
    plt.title('Hybrid Confusion Matrix (Best Run)')
    plt.show()
