

# bart_out = open("bart-baseline-2ep.txt", "r")
prompts = open("val_prompts.txt", "r")
stories = open("ant_data_2.txt", "r")
union = open("test_results_probabilities_2.txt", "r")
prompt_lines = prompts.readlines()
story_lines = stories.readlines()
union_scores = union.readlines()

print(len(prompt_lines))
print(len(story_lines))
print(len(union_scores))
# inspect specific story
i = 99
print(prompt_lines[i])
print(story_lines[i])
print("Union score: ", union_scores[i])

# print(len(bart_out.readlines()))
# print(len(prompts.readlines()))
# print(len(human_out.readlines()))
