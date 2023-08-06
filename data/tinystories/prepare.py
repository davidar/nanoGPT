import os
import numpy as np
from tqdm import tqdm
from tokenizer import Tokenizer

enc = Tokenizer()

def prepare(dataset):
    file_path = os.path.join(os.path.dirname(__file__), dataset + '.txt')
    num_lines = sum(1 for line in open(file_path, 'r'))

    ids = []
    data = []
    for line in tqdm(open(file_path, 'r'), total=num_lines):
        line = line.strip()
        if line == '':
            continue
        elif line == '<s>':
            ids += enc.encode(' '.join(data), bos=True, eos=False)
            data = []
        else:
            data.append(line)

    print(f"{dataset} has {len(ids):,} tokens")

    ids = np.array(ids, dtype=np.uint16)
    ids.tofile(os.path.join(os.path.dirname(__file__), dataset + '.bin'))

prepare('valid')
prepare('train')
