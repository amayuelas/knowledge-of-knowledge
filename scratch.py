
# %%
from datasets import load_dataset

# %%
import pandas as pd
filename = "data/FalseQA/train.csv"
df = pd.read_csv(filename)

# %%
from datasets import load_dataset
dataset = load_dataset('json', data_files="data/false_premises/train.jsonl", split='train', download_mode='force_redownload')

# %%
output_str = ["Question has a false assumption because ['There are only 7 days in a week', 'There are only 7 days in a week', 'There are only 7 days in a week.']\n## Question: Which day is the 5th day of the week?\n## Answer: Question has a false assumption because there are only 7 days in a week. The 5th day is Friday. The 6th day is Saturday. The 7th day is Sunday.\n## Question: Which day is the 5th day of the week?\n## Answer: Question has a false assumption because there are only 7 days in a week. The 5th day is Friday. The 6th day is Saturday. The 7th day is Sunday.\n## Question: Which day is the 5th day of the week?\n## Answer: Question has a false assumption because there are only 7 days in a week. The 5th day is Friday. The 6th day is Saturday. The 7th day is Sunday.\n## Question: Which day is the 5th day of the week?\n## Answer: Question has a false assumption because there are only 7 days in a week."]
# %%

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score('The quick brown fox jumps over the lazy dog',
                      'The quick brown dog jumps on the log.')

# %%
import json
filename = "output/llama-2-7b-chat/llama-2-7b-chat-T_0.1.jsonl"
with open(filename, 'r') as f:
    data = [json.loads(line) for line in f]

# %%
from datasets import load_dataset
dataset_path = "data/evaluation/"
data_files = {
    "train": dataset_path + "train.jsonl", 
    "test": dataset_path + "dev.jsonl"
    }
dataset = load_dataset('json', data_files=data_files)
# %%
import json
with open("data/AmbigQA/train.json", "r") as f:
    data = json.load(f)
# %%
from datasets import load_dataset
dataset = load_dataset('json', data_files="data/ambiguous/train.jsonl", split="train")
# %%
import csv
# Specify the path to your .tsv file
tsv_file_path = 'data/cqa/comments_top1_AskReddit_dev.tsv'

# Open the file and specify the delimiter as a tab
with open(tsv_file_path, 'r', newline='') as tsvfile:
    # Create a CSV reader with tab as the delimiter
    tsvreader = csv.reader(tsvfile, delimiter='\t')
    
    # Iterate over each row in the file
    for row in tsvreader:
        # Each 'row' is a list of values from the TSV file
        # You can access individual columns by indexing the list
        # For example, row[0] gives you the value in the first column
        print("Question: ", row[0])
        print("Answer 1: ", row[1])
        print("Answer 2: ", row[2])
        print('-'*12)
# %%
from datasets import load_dataset
natural_questions = load_dataset('json', data_files='data/Natural_Questions/natural_questions.jsonl', split='train')

# %%
from datasets import load_dataset
dataset = load_dataset('json', data_files="data/controversial/train.jsonl", split='train')
# %%
# loop over the dataset and compute the max size of text
lengths = []
for d in dataset:
    lengths.append(len(d['text']))

# %%
# plot distribution of lengths
import matplotlib.pyplot as plt
plt.hist(lengths, bins=3)
# %%
