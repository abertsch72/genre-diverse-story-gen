from typing import Text, Dict


def load_prompts() -> (Dict[Text: Text], Dict[Text: Text]):
    # prompt and story values should both be lists
    train_data = {"prompt": None, "story": None}
    val_data = {"prompt": None, "story": None}

    assert len(train_data["prompt"]) == len(train_data["story"])
    assert len(val_data["prompt"]) == len(val_data["story"])

    return train_data, val_data
