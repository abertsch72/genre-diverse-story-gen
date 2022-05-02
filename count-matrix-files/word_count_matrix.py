import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import os
import pickle

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

max_input_id = 50257

# genres
''' MAPPINGS
categories = {"horror": 883, "fantasy": 1206, "romance": 1235,
              "science fiction": 1213, "adventure": 892, "sports": 1126,
              "western": 871, "humor & comedy": 882, "children's": 61, "urban": 873,
              "thriller & suspense": 874, "religious": 877, "YA/teen": 1018, "mystery & detective": 879}
'''

genre_folders = ["1206", "1235", "1213", "1126",
                 "1018", "892", "883", "882",
                 "879", "877", "874", "873", "871", "61"]

# create count matrix where rows are the GPT-2 token ids, cols are the genres from genre_folders
# last col is the total count of the word w over all the genres

'''
count_matrix = np.zeros((max_input_id, len(genre_folders)+1))
d = dict()  # keys are file names, value is input ids array
genre_index = 0
for genre_name in genre_folders:
    print(genre_name)
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

last col is the total count of word w across all genres
count_matrix[:, len(genre_folders)] = np.sum(count_matrix, axis=1)


print("done!:)")
np.save('count-matrix.npy', count_matrix)

'''


# get total word counts in each genre
'''
total_genre_counts = np.sum(X, axis=0)
np.save('total-genre-counts.npy', total_genre_counts)
'''


# count number of unique tokens in each genre
'''
X = np.load('count-matrix.npy')
genre_unique_counts = np.count_nonzero(X, axis=0)
np.save('genre-unique-counts.npy', genre_unique_counts)
'''

# count total number of unique tokens across all genres
'''
X = np.load('count-matrix.npy')
print(50257 - len(np.where(~X.any(axis=1))[0]))
'''
