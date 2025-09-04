import nltk
from nltk.tag import HiddenMarkovModelTagger
from nltk.corpus import treebank
from nltk import word_tokenize

try:
    nltk.data.find('corpora/treebank')
except nltk.downloader.DownloadError:
    nltk.download('treebank')
    
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

corpus = treebank.tagged_sents()
train_data = corpus[:3000]
test_data = corpus[3000:]

hmm_tagger = HiddenMarkovModelTagger.train(train_data)

sentence = input("Enter a sentence: ")

tokens = word_tokenize(sentence)
tagged_sentence = hmm_tagger.tag(tokens)

print(tagged_sentence)
