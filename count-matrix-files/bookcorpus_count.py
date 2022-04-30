import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import os
import pickle
from datasets import load_dataset

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

max_input_id = 50257
genre_folders = ["1206", "1235"]

corpus = load_dataset("bookcorpus")
print(corpus)
count_matrix = np.zeros((max_input_id, 1))
d = dict()  # keys are file names, value is input ids array
text = corpus['train']["text"]
stepsize = 1000
for subset in range(0, len(text), stepsize):
    subtext = text[subset, min(subset+stepsize, len(text))]
    print(len(subtext))
    token_ids = tokenizer(subtext)["input_ids"]
    id_arr = np.array(token_ids)
    unique_ids = set(token_ids)

    for id in unique_ids:
        count_matrix[id] += np.count_nonzero(id_arr == id)
    print(subset)

# with open(f'learning_rate={learning_rate}.pickle', 'rb') as handle:
#             d = pickle.load(handle)

print("done!:)")
np.save('count-matrix.npy', count_matrix)

# X = np.load('count-matrix.npy')

# get total word counts in each genre
# total_genre_counts = np.sum(X, axis=0)
# np.save('total-genre-counts.npy', total_genre_counts)

# genre_counts = np.load("total-genre-counts.npy")
# print(genre_counts)
