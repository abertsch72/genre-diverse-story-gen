# Credit to: https://www.kaggle.com/code/ratthachat/writingprompts-combine-one-line-data-for-gpt2/notebook


def load_prompts(num_words = None, dir = "data/"):

    # num_words: The number of words to be used from each story (when None is passed full stories are returned)
    # dir: Directory of folder containing prompts
    
    directories = ["train", "test", "valid"]
    data = dict()

    for name_id in range(len(directories)):

        fp = open(dir + directories[name_id] + ".wp_source") 
        ft = open(dir + directories[name_id] + ".wp_target") 

        
        prompts = fp.readlines()
        stories = ft.readlines()

        assert(len(stories) == len(prompts))

        if (num_words != None):
            stories = [prompts[i].rstrip()+ " <endprompts> " + " ".join(stories[i].split()[0:num_words]) for i in range(len(stories))]

        data[directories[name_id]] = {"prompt": prompts, "story": stories}
        
    return data["train"], data["valid"], data["test"],

