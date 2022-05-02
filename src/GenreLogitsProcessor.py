from transformers.generation_logits_process import LogitsProcessor
import torch
import numpy as np
from typing import Text


class InfluenceGenreLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] weighting model probabilities by conditional likelihood in a given genre.
    Args:
        repetition_penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """

    def __init__(self, genre: Text, method: Text = "MAP", lambda_val: float = 0.1):
        count_matrix = np.load("./../count-matrix-files/count-matrix.npy")
        total_genre_counts = np.load("./../count-matrix-files/total-genre-counts.npy")
        bookcorp_count = None
        bookcorp_total_count = None
        valid_genres = ["fantasy", "romance", "scifi", "sports", "YA", "adventure", "horror", "comedy", "mystery", "religious", "thriller", "urban", "western", "children's"]
        valid_methods = ["MLE", "MAP"]
        if genre not in valid_genres:
            raise ValueError(f"Genre {genre} is not in list of valid genres: {valid_genres}")
        if method not in valid_methods:
            raise ValueError(f"Method {method} is not in list of valid methods: {valid_methods}")

        self.genre = genre
        self.genre_index = valid_genres.index(self.genre)
        if method == "MLE":
            self.probabilities = count_matrix[:, self.genre_index]/total_genre_counts[self.genre_index] # TODO: load in the correct list of probabilities, based on genre and method
        else: #MAP
            alpha = # count of tok1  -- orig bookcorp
            beta = # total token count - alpha   -- orig bookcorp
            self.probabilities = (count_matrix[:, self.genre_index] + alpha - np.ones((50257, 1))/
                                total_genre_counts[self.genre_index] + alpha + beta - np.full((50257, 1), 2))

        assert self.probabilities.shape == torch.size([1, 50257])
        # TODO: make our scores log probabilities

        self.lambda_val = lambda_val #TODO: is the default I've provided (0.1) the best value?


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # shape of scores is torch.Size([1, 50257]); they appear to be un-calibrated negative scores (so make ours log probs too)

        # TODO: combine our probabilities with the scores, using the formula from our presentation

        return scores
