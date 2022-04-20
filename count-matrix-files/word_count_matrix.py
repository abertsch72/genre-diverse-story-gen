import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import os
import pickle

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

max_input_id = 50257
genre_folders = ["1206", "1235"]

count_matrix = np.zeros((max_input_id, len(genre_folders)))
d = dict()  # keys are file names, value is input ids array
genre_index = 0
for genre_name in genre_folders:
    for filename in os.listdir("data/" + genre_name):
        print(filename)
        with open(os.path.join("data/" + genre_name, filename), 'r', errors='ignore') as f:
            text = f.read()

            token_ids = tokenizer(text)["input_ids"]
            d[filename] = token_ids

            id_arr = np.array(token_ids)
            unique_ids = set(token_ids)

            for id in unique_ids:
                count_matrix[id, genre_index] += np.count_nonzero(id_arr == id)

    genre_index += 1

with open('tok_ids.pickle', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
