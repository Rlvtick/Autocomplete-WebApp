# Install Streamlit

""" !pip install streamlit -q """

# Generate Brown Dataset

import nltk
from nltk.corpus import brown
nltk.download('brown')

# Get all the text from the Brown Corpus
data_teks = ' '.join(brown.words())

# Data Preprocessing 
data_teks = data_teks.lower() 

with open('brown.txt', "w") as file_txt:
    file_txt.write(data_teks)

# Streamlit app.py File

%%writefile app.py
 
import streamlit as st
import re
import string
from nltk.util import ngrams
from nltk.lm import MLE
from nltk.lm.preprocessing import pad_sequence
import random
import math
nltk.download('punkt')
 
 # N-Grams Function 
def build_ngrams(token, n):
    ngrams = {}
    for sentence in token:
        words = sentence.split()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i + n - 1])
            next_word = words[i + n - 1]
            if ngram in ngrams:
                ngrams[ngram].append(next_word)
            else:
                ngrams[ngram] = [next_word]
    return ngrams
 
 # Autocomplete Function
def autocomplete(input_text, ngrams, n):
    words = input_text.split()
    while len(words) < n:
        ngram = tuple(words[-(ngram_size - 1):])
        if ngram in ngrams:
            next_word = ngrams[ngram][0]
            words.append(next_word)
        else:
            break
    return " ".join(words)
 
# MLE Function
def autocomplete_mle(input_text, max_words, model):
    input_tokens = nltk.word_tokenize(input_text.lower())
    context = pad_sequence(input_tokens, ngram_size, pad_left=True, pad_right=False)
    predictions = model.generate(num_words=max_words, text_seed=context)

    # Remove punctuation from predictions
    predictions_without_punctuation = [
        word for word in predictions if word not in string.punctuation]

    # Combine words into a single text string
    predicted_text = ' '.join(predictions_without_punctuation)
    return predicted_text

 # Visualize Streamlit Application
def main():
    # The data processing is only done once when the application starts 
    if 'ngrams_simple' not in st.session_state:

      # Building the N-Grams Model
      ngrams_simple = build_ngrams(tokenBySentences, ngram_size)

    if 'model' not in st.session_state:
      # Initializing the MLE Model
      model = MLE(ngram_size)
      ngrams_data = list(ngrams(tokenByWords, ngram_size))
      model.fit([ngrams_data], vocabulary_text=tokenByWords)

    st.title("Autocomplete dengan N-Grams")
    input_text = st.text_input("Masukkan teks:", "my father was")
    n_kata = st.number_input("banyak kata yang ingin diprediksi:", 7)

    if st.button("Autocomplete"):
        hasil_autocomplete = autocomplete(input_text, ngrams_simple, n_kata)
        st.write("Hasil Model N-Grams:")
        st.write(hasil_autocomplete)
 
        hasil_autocomplete_mle = autocomplete_mle(input_text, n_kata, model)
        st.write("Hasil Autocomplete:")
        st.write(input_text,hasil_autocomplete_mle)
 
if __name__ == "__main__":
    with open('corpus.txt', "r", encoding="utf-8") as file_txt:
      text = file_txt.read()
    text = text[:len(text) // 5]

    # Reading the text and creating tokens
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s.]', '', text)
    text = re.sub(r'[\n]', '', text)
    tokenByWords = nltk.word_tokenize(text)
    tokenBySentences = text.split('. ')
 
    # The N-Grams size that users wants to use
    ngram_size = 3

    # Run the Streamlit Application
    main()

%%writefile app.py
 
import nltk
import streamlit as st
import re
import string
from nltk.util import ngrams
from nltk.lm import MLE
from nltk.lm.preprocessing import pad_sequence
import random
import math
nltk.download('punkt')
 
# Fungsi untuk membangun n-grams
def build_ngrams(token, n):
    ngrams = {}
    for sentence in token:
        words = sentence.split()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i + n - 1])
            next_word = words[i + n - 1]
            if ngram in ngrams:
                ngrams[ngram].append(next_word)
            else:
                ngrams[ngram] = [next_word]
    return ngrams
 
# Fungsi untuk autocompletion
def autocomplete(input_text, ngrams, n):
    words = input_text.split()
    while len(words) < n:
        ngram = tuple(words[-(ngram_size - 1):])
        if ngram in ngrams:
            next_word = ngrams[ngram][0]
            words.append(next_word)
        else:
            break
    return " ".join(words)
 
# fungsi mle
def autocomplete_mle(input_text, max_words, model):
    input_tokens = nltk.word_tokenize(input_text.lower())
    context = pad_sequence(input_tokens, ngram_size, pad_left=True, pad_right=False)
    predictions = model.generate(num_words=max_words, text_seed=context)

    # Remove punctuation from predictions
    predictions_without_punctuation = [
        word for word in predictions if word not in string.punctuation]

    # Combine words into a single text string
    predicted_text = ' '.join(predictions_without_punctuation)
    return predicted_text

with open('brown.txt', "r", encoding="utf-8") as file_txt:
  text = file_txt.read()
#text = text[:len(text) // 20]

# Baca teks dan buat token
text = text.lower()
text = re.sub(r'[^a-zA-Z\s.]', '', text)
text = re.sub(r'[\n]', '', text)
tokenByWords = nltk.word_tokenize(text)
tokenBySentences = text.split('. ')
 
# Ukuran n-grams yang ingin Anda gunakan
ngram_size = 3
 
# Bangun model n-grams
ngrams_simple = build_ngrams(tokenBySentences, ngram_size)
 
# Inisialisasi model MLE
model = MLE(ngram_size)
ngrams_data = list(ngrams(tokenByWords, ngram_size))
model.fit([ngrams_data], vocabulary_text=tokenByWords)
 
# Tampilan aplikasi Streamlit
st.title("Autocomplete berbasis N-Grams dan MLE")
input_text = st.text_input("Masukkan teks:", "she was a")
n_kata = st.number_input("banyak kata yang ingin diprediksi:", 0)
 
if st.button("Autocomplete"):
  hasil_autocomplete = autocomplete(input_text, ngrams_simple, n_kata+3)
  st.write("Hasil Model N-Grams:")
  st.write(hasil_autocomplete)
 
  hasil_autocomplete_mle = autocomplete_mle(input_text, n_kata, model)
  st.write("Hasil Model MLE:")
  st.write(input_text,hasil_autocomplete_mle)

""" !wget -q -O - ipv4.icanhazip.com """

""" !streamlit run app.py & npx localtunnel --port 8501 """