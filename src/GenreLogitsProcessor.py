from transformers.generation_logits_process import LogitsProcessor
from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np
from typing import Text

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

'''
python3 generation.py --model_type gpt2 --model_name_or_path gpt2 --prompt "I wish I could " --genre "fantasy"

python3 generation.py --model_type gpt2 --model_name_or_path src/short-story-model --prompt "Once upon a time, " --genre "fantasy" --method "MLE"  --lambda_val 0.6

'''

class InfluenceGenreLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] weighting model probabilities by conditional likelihood in a given genre.
    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    def __init__(self, genre: Text, method: Text = "MAP", lambda_val: float = 0.00001, scale: int = 14):
        count_matrix = np.load("./../count-matrix-files/count-matrix.npy")
        total_genre_counts = np.load("./../count-matrix-files/total-genre-counts.npy")
        bookcorp_count = np.load("./../count-matrix-files/bookcorpus-counts.npy")
        bookcorp_total_count = np.sum(bookcorp_count)
       

        valid_genres = ["fantasy", "romance", "scifi", "sports", "YA", "adventure", "horror", "comedy", "mystery", "religious", "thriller", "urban", "western", "children's"]
        valid_methods = ["MLE", "MAP", "newMLE"]
        if genre not in valid_genres:
            raise ValueError(f"Genre {genre} is not in list of valid genres: {valid_genres}")
        if method not in valid_methods:
            raise ValueError(f"Method {method} is not in list of valid methods: {valid_methods}")

        self.genre = genre
        self.genre_index = valid_genres.index(self.genre)
        if method == "MLE":
            self.probabilities = count_matrix[:, self.genre_index]/total_genre_counts[self.genre_index] # divided by token over all genres TODO: load in the correct list of probabilities, based on genre and method
            # ind = np.argpartition(self.probabilities.reshape((50257,)), -500)[-500:-200]
            # ind =  ind[np.argsort(self.probabilities[ind])]
            #print([tokenizer.decode(x) for x in ind])

        elif method == "newMLE":
            num = count_matrix[:, self.genre_index]/total_genre_counts[self.genre_index]
            denom = count_matrix[:, -1]/ np.sum(total_genre_counts)
            self.probabilities = np.where(denom == 0, 0, num/denom)
            self.probabilities = self.probabilities / np.sum(self.probabilities)
            ind = np.argpartition(self.probabilities.reshape((50257,)), -100)[-100:]
            ind =  ind[np.argsort(self.probabilities[ind])]
            print([tokenizer.decode(x) for x in ind])
            print(self.probabilities[ind])
        
        else: #MAP
            #MLE
            num = count_matrix[:, self.genre_index]/total_genre_counts[self.genre_index]
            denom = count_matrix[:, -1]/ np.sum(total_genre_counts)
            mle = np.where(denom == 0, 0, num/denom)
            mle = mle / np.sum(mle)

            alpha = bookcorp_count.reshape((50257,)) / scale # count of tok1  -- orig bookcorp
            beta = (np.full((50257,), bookcorp_total_count) - alpha) / scale # total token count - alpha   -- orig bookcorp
            denom1 = (total_genre_counts[self.genre_index] + alpha + beta - np.full((50257,), 2))
            num1 = (count_matrix[:, self.genre_index] + alpha - np.ones((50257,)))
            num = np.where(denom1 == 0, num1, num1/denom1)

            # denom2 = (np.sum(total_genre_counts) + alpha + beta - np.full((50257,), 2))
            # num2 = (count_matrix[:, -1] + alpha - np.ones((50257,)))
            # denom =  np.where(denom2 == 0, num2, num2/denom2)  
            # self.probabilities = np.where(denom==0, num, num/denom)
            
            self.probabilities = np.where(mle == 0, num, num / mle)
            self.probabilities = np.where(self.probabilities < 0, 0, self.probabilities)
            self.probabilities = self.probabilities / np.sum(self.probabilities)
    

            ind = np.argpartition(self.probabilities.reshape((50257,)), -100)[-100:]
            ind =  ind[np.argsort(self.probabilities[ind])]
            print([tokenizer.decode(x) for x in ind])
            print(self.probabilities[ind])

        assert self.probabilities.shape == (50257,)
        # TODO: make our scores log probabilities

        self.lambda_val = lambda_val #TODO: is the default I've provided (0.1) the best value?


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # shape of scores is torch.Size([1, 50257]); they appear to be un-calibrated negative scores (so make ours log probs too)

        # TODO: combine our probabilities with the scores, using the formula from our presentation
        probabilities = torch.from_numpy(self.probabilities.reshape((1, 50257)))
        scores = self.lambda_val * scores + (1-self.lambda_val) * np.log(probabilities)

        return scores
