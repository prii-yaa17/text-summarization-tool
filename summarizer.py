import nltk
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def summarize_text(text, num_sentences=4):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)

    ranked_sentence_indices = np.argsort(sentence_scores)[::-1]

    selected_indices = sorted(ranked_sentence_indices[:num_sentences])

    summary = " ".join([sentences[i] for i in selected_indices])

    return summary


if __name__ == "__main__":
    print("\n====== TEXT SUMMARIZATION TOOL ======\n")
    print("Paste your article below.")
    print("After pasting, press CTRL+Z and then ENTER.\n")

    article = sys.stdin.read()

    if len(article.strip()) == 0:
        print("No text provided.")
        sys.exit()

    print("\nGenerating Summary...\n")

    summary = summarize_text(article)

    print("\n====== SUMMARY ======\n")
    print(summary)