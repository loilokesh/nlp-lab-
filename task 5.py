import nltk
from nltk.util import ngrams
from nltk.lm import Laplace
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline

# Download required NLTK data
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('punkt')

def ngram_smoothing(sentence, n):
    # Tokenize the sentence and convert to lowercase
    tokens = word_tokenize(sentence.lower())
    
    # Prepare data for language model: input must be a list of tokenized sentences
    train_data, padded_sents = padded_everygram_pipeline(n, [tokens])
    
    # Create and train the Laplace-smoothed model
    model = Laplace(n)
    model.fit(train_data, padded_sents)

    return model, tokens

# Take input from the user
sentence = input("Enter a sentence: ")
n = int(input("Enter the value of N for N-grams: "))

# Build the model and get the tokenized input
model, tokens = ngram_smoothing(sentence, n)

# Determine the context (last n-1 tokens)
if len(tokens) < n - 1:
    context = tokens
else:
    context = tokens[-(n - 1):]

# Generate next 3 words based on the context
generated_words = []
for _ in range(3):
    next_word = model.generate(1, text_seed=context)
    generated_words.append(next_word)
    
    # Update context by adding the new word and keeping the last n-1 tokens
    context = (context + [next_word])[-(n - 1):]

print("Next words:", ' '.join(generated_words))
