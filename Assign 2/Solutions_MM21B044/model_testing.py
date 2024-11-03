import numpy as np
import os, csv, re, pickle

def load_model():
    global sm_prob, hm_prob, p_spam, p_ham, tot_sm_words, tot_hm_words, vocab
    with open('spam_ham_model.pkl', 'rb') as model_file:
        (sm_prob, hm_prob, p_spam, p_ham, tot_sm_words, tot_hm_words, vocab) = pickle.load(model_file)

def make_usable(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)    # Remove special characters and digits
    return text.split()

def predict_email(words, alp = 1):
    """Predicts if an email is spam (+1) or ham (0) based on the words in the email."""
    spam_score = np.log(p_spam)
    ham_score = np.log(p_ham)

    for word in words:
        if word in vocab:
            spam_score += np.log(sm_prob.get(word, alp / (tot_sm_words + alp * len(vocab))))
            ham_score += np.log(hm_prob.get(word, alp / (tot_hm_words + alp * len(vocab))))

    return 1 if spam_score > ham_score else 0

def test_spam(folder_path='test', output_csv='predictions.csv'):
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as file:
                email_content = file.read()
                words = make_usable(email_content)
                prediction = predict_email(words)
                results.append((filename, prediction))

    # Write results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Prediction'])
        writer.writerows(results)

    print(f"Predictions saved to {output_csv}")

load_model()
test_spam()