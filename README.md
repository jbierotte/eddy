import re
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import nltk

# Download NLTK tokenizer resources (run once)
nltk.download('punkt')

# Step 1: Sample Kalenjin Sentences
kalenjin_sentences = [
    "Kiptaiyat ak muren eng kisumet.",
    "Koee inendet ab kasit ne kikoomi.",
    "Kimnai lagok che mi konom kongoi.",
    "Kapkutuny nebo boiyot komwa.",
    "Amun kalyet ne bo kotik kiptendeny."
]

# Step 2: Preprocess the Text
def preprocess_text(sentences):
    """
    Clean and tokenize sentences.
    """
    clean_sentences = []
    for sentence in sentences:
        # Remove punctuation and convert to lowercase
        sentence = re.sub(r"[^\w\s]", "", sentence).lower()
        # Tokenize the sentence
        tokens = word_tokenize(sentence)
        clean_sentences.append(tokens)
    return clean_sentences

# Apply preprocessing
preprocessed_corpus = preprocess_text(kalenjin_sentences)
print("Preprocessed Corpus:", preprocessed_corpus)

# Step 3: Train Word2Vec Model
model = Word2Vec(
    sentences=preprocessed_corpus,  # Preprocessed data
    vector_size=100,                # Dimensionality of the word vectors
    window=5,                       # Context window size
    min_count=1,                    # Minimum frequency for words to be included
    workers=4                       # Number of CPU cores for training
)

# Save the model
model.save("kalenjin_word2vec.model")
print("Word2Vec model saved as 'kalenjin_word2vec.model'.")

# Step 4: Test the Model
# Get the vector for a specific word
word_vector = model.wv['kiptaiyat']  # Example word
print("\nVector for 'kiptaiyat':\n", word_vector)

# Find similar words
similar_words = model.wv.most_similar('kiptaiyat')  # Example word
print("\nWords similar to 'kiptaiyat':", similar_words)

# Step 5: Expand the Corpus (Optional)
# Uncomment the following lines to load text from a file and preprocess it:
# with open('kalenjin_text.txt', 'r', encoding='utf-8') as file:
#     text_data = file.readlines()
# preprocessed_corpus = preprocess_text(text_data)
# print("Preprocessed Large Corpus:", preprocessed_corpus)

