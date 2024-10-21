import numpy as np

with open('1268-0.txt', 'r', encoding='utf-8') as fp:
    text = fp.read()
start_indx = text.find('THE MYSTERIOUS ISLAND')
end_indx = text.find('End of the Project Gutenberg')
text = text[start_indx:end_indx]
char_set = set(text)
print('Total Length:', len(text))
print('Unique Characters:', len(char_set))
chars_sorted = sorted(char_set)
char2int = {ch:i for i,ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)
text_encoded = np.array([char2int[ch] for ch in text], dtype=np.int32)
print('Text encoded shape:', text_encoded.shape)
print(text[:15], '== Encoding ==>', text_encoded[:15])
print(text_encoded[15:21], '== Reverse ==>', ''.join(char_array[text_encoded[15:21]]))
for ex in text_encoded[:5]:
    print('{} -> {}'.format(ex, char_array[ex]))

import torch
from torch.utils.data import Dataset
seq_length = 40
chunk_size = seq_length + 1
text_chunks = [text_encoded[i:i+chunk_size] for i in range(len(text_encoded)-chunk_size)]
# print(text_encoded.shape)
# print(len(text_chunks))
from torch.utils.data import Dataset
class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks
    def __len__(self):
        return len(self.text_chunks)
    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()

seq_dataset = TextDataset(torch.tensor(text_chunks))
for i, (seq, target) in enumerate(seq_dataset):
    print(' Input (x): ', repr(''.join(char_array[seq])))
    print(' Target (y): ', repr(''.join(char_array[target])))
    print()
    if i == 1:
        break