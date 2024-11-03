## Nikshay Jain | MM21B044
### DA5400 - FML: Assign 2

import re, os
import pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc

data = pd.read_csv('enron_spam_data.csv')

# EDA + preprocessing
print("Ratio of spam:ham mails in dataset is: ", sum(data['Spam/Ham']=='spam')/sum(data['Spam/Ham']=='ham'))   # ratio of spam:ham

data = data.drop(['Message ID', 'Date'], axis=1)
data['Spam/Ham'].replace({'spam': 1, 'ham': 0}, inplace=True)
data = data.rename(columns={'Spam/Ham': 'label'})
data.fillna("", inplace = True)
data['Spam/Ham'] = 1 if "spam" else 0

def make_usable(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)    # Remove special characters and digits
    return text.split()

data['body'] = (data['Message']).apply(make_usable)

def split(X, y, test_size=0.2):
    num_samples = len(X)
    
    indices = np.random.permutation(num_samples)
    
    num_test_samples = int(test_size * num_samples)
    
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]
    
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split(data['body'], data['label'], test_size=0.2)

# Get word frequencies for each class
sp_word = []
hm_word = []

for words, label in zip(X_train, y_train):
    if label == 1:
        sp_word.extend(words)
    else:
        hm_word.extend(words)

sp_word_counts = Counter(sp_word)
hm_word_counts = Counter(hm_word)

# Calculate probabilities
total_spam_words = sum(sp_word_counts.values())
total_ham_words = sum(hm_word_counts.values())
vocab = list(set(sp_word_counts.keys()).union(set(hm_word_counts.keys())))
vocab_size = len(vocab)

# Calculate prior probabilities for each class
p_spam = y_train.mean()
p_ham = 1 - p_spam

# Laplace smoothing
alpha = 1
spam_word_probs = {word: (sp_word_counts[word] + alpha) / (total_spam_words + alpha * vocab_size) for word in vocab}
ham_word_probs = {word: (hm_word_counts[word] + alpha) / (total_ham_words + alpha * vocab_size) for word in vocab}

def predict(words):
    spam_score = np.log(p_spam)
    ham_score = np.log(p_ham)
    vocab_set = set(vocab)

    # Precompute log probabilities for known words
    for word in words:
        if word in vocab_set:  # Check membership in the set
            spam_word_prob = spam_word_probs.get(word, alpha / (total_spam_words + alpha * vocab_size))
            ham_word_prob = ham_word_probs.get(word, alpha / (total_ham_words + alpha * vocab_size))
            spam_score += np.log(spam_word_prob)
            ham_score += np.log(ham_word_prob)

    return 1 if spam_score > ham_score else 0

# Predict on test set
y_pred = X_test.apply(predict)

# Evaluate over metrics
accuracy = (y_pred == y_test).mean()
f1 = f1_score(y_pred,y_test)
print(f'f1-score: {f1:.4f}')
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, thr = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Save the trained model to a file
with open('spam_ham_model.pkl', 'wb') as model_file:
    pickle.dump((spam_word_probs, ham_word_probs, p_spam, p_ham, total_spam_words, total_ham_words, vocab), model_file)

print("Model exported as:", model_file)