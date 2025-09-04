
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
text = "Tokenization without transformers is straightforward with tools like NLTK."
tokens = word_tokenize(text)
print("Tokens:", tokens)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens_transformers = tokenizer(text, return_tensors="pt")
print("Transformers Tokens:", tokens_transformers)
tokens_transformers_list = tokenizer.convert_ids_to_tokens(tokens_transformers['input_ids'][0].tolist())
print("Transformers Tokens (List):", tokens_transformers_list)
decoded_text = tokenizer.decode(tokens_transformers['input_ids'][0], skip_special_tokens=True)
print("Decoded Text:", decoded_text)
