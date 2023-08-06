import sys
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

for line in open(sys.argv[1]):
    line = line.strip()
    line = line.lower()
    if line == '<|endoftext|>':
        print('<s>')
    else:
        print(' '.join(word_tokenize(line)))
