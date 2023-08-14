import os
import numpy as np
from tokenizer import Tokenizer

train_data = np.memmap(os.path.join(os.path.dirname(__file__), "train.bin"), dtype=np.uint16, mode="r")
text = train_data[:1000].tolist()
print(Tokenizer().decode(text))
