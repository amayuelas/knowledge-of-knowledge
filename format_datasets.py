import os
import ast
import csv
import json
import random
import argparse
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

paths = {
        "qaqa": ["data/QAQA_Dec2022/QAQA_evaluation_set_Dec2022.csv"],
        "FalseQA": ["data/FalseQA/train.csv", "data/FalseQA/test.csv", "data/FalseQA/valid.csv"],
        "AmbigQA": ["data/AmbigQA/train.json", "data/AmbigQA/dev.json"],
        "cqa": ["data/cqa/comments_top1_AskReddit_train.tsv", "data/cqa/comments_top1_AskReddit_dev.tsv", "data/cqa/comments_top1_AskReddit_test.tsv"],
        "knowledge-of-knowledge": ["data/knowledge-of-knowledge/knowns_unknowns.jsonl"]
    }


def format_qaqa(dataset_paths):

    df = pd.read_csv(dataset_paths[0])
    for i in range(1, len(dataset_paths)):
        df = pd.concat([df, pd.read_csv(dataset_paths[i])])

    texts = []
    for i in range(len(df)):
        if pd.isna(df.iloc[i]['questionable_assumption']):
            question = df.iloc[i]["question"]
            answer = df.iloc[i]["abstractive_answer"]
            text = "### Question: " + question +  "\n### Answer: " + answer
            line = {"question": question, "answer": [answer], "text": text, "source": "QAQA", "label": 0}
        else:
            question = df.iloc[i]["question"]
            answer = df.iloc[i]["abstractive_answer"]
            text = "### Question: " + question +  "\n### Answer: Question has a questionable assumption because " + answer[0].lower() + answer[1:]
            line = {"question": question, "answer": [answer], "text": text, "source": "QAQA", "label": 1}

        texts.append(line)

    return texts


def format_FalseQA(dataset_paths):
    
    df = pd.read_csv(dataset_paths[0])
    for i in range(1, len(dataset_paths)):
        df = pd.concat([df, pd.read_csv(dataset_paths[i])])

    texts = []
    for i in range(len(df)):
        if df.iloc[i]['label'] == 0:
            question = df.iloc[i]["question"]
            answer = df.iloc[i]["answer"]
            text = "### Question: " + question +  "\n### Answer: " + answer
            line = {"question": question, "answer": [answer], "text": text, "source": "FalseQA", "label": 0}
        else:
            question = df.iloc[i]["question"]

            answers = df.iloc[i]['answer']
            if answers[0] == '[':
                answers = ast.literal_eval(answers)
                answer = random.choice(answers)
            else:
                answer = answers
            
            text = "### Question: " + df.iloc[i]["question"] +  "\n### Answer: Question has a false assumption because " + answer[0].lower() + answer[1:]
            line = {"question": question, "answer": [answer], "text": text, "source": "FalseQA", "label": 1}
        texts.append(line)
    
    return texts


def format_AmbigQA(dataset_paths):

    data = []
    for dataset_path in dataset_paths:
        with open(dataset_path, "r") as f:
            data.extend(json.load(f))

    texts = []
    for i in range(len(data)):
        question = data[i]['question']
        answers = data[i]['nq_answer']
        if len(answers) == 1:
            text = "### Question: " + question + "\n### Answer: " + answers[0]
            label = 0
        elif len(answers) == 2:
            text = "### Question: " + question + "\n### Answer: Question is ambiguous because the answer could be " + answers[0] + " or " + answers[1]
            label = 1
        else:
            text = "### Question: " + question + "\n### Answer: Question is ambiguous because the answer could be " + ", ".join(answers[:-1]) + ", or " + answers[-1]
            label = 1

        data[i]['text'] = text
        line = {"question": question, "answer": answers, "text": text, "source": "AmbigQA", "label": label}
        texts.append(line)

    # Balance dataset based on label
    random.shuffle(texts)
    label_0 = [line for line in texts if line['label'] == 0]
    label_1 = [line for line in texts if line['label'] == 1]
    if len(label_0) > len(label_1):
        label_0 = label_0[:len(label_1)]
    else:
        label_1 = label_1[:len(label_0)]
    texts = label_0 + label_1

    return texts


def format_cqa(dataset_paths):

    data = []
    for dataset_path in dataset_paths:
        with open(dataset_path, 'r', newline='') as tsvfile:
            # Create a CSV reader with tab as the delimiter
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            data.extend([row for row in tsvreader])
    
    # Iterate over each row in the file
    texts= []
    for row in data:
        question = row[0]
        # It is not clear which is the best answer and the most controversial one. 
        # For now, we will take 
        # - answer1: the most controversial one 
        # - answer2: the best answer.
        answer1 = row[1] 
        answer2 = row[2]

        text = "### Question: " + question + "\n### Answer: Question is controversial. One could say " + answer2
        line = {"question": question, "answer": [answer2], "text": text, "source": "cqa", "label": 1}
        texts.append(line)

    # Retrieve the same number of queries from Natual Questions dataset
    natural_questions = load_dataset('json', data_files='data/Natural_Questions/natural_questions.jsonl', split='train')
    natural_questions = natural_questions.shuffle()
    natural_questions = natural_questions.select(range(len(texts)))
    for q in natural_questions:
        question = q['question']
        if random.randint(0, 1) == 0:
            answer = q['short_answer']
            if answer == None:
                answer = q['long_answer']
        else:
            answer = q['long_answer']

        # TO BE REMOVED
        answer = q['short_answer']
        if answer == None:
            continue

        text = "### Question: " + question + "\n### Answer: " + answer
        line = {"question": question, "answer": [answer], "text": text, "source": "cqa", "label": 0}
        texts.append(line)

    # Sufffle the dataset
    random.shuffle(texts)

    return texts


def format_knowledge_of_knowledge(dataset_paths):

    data = load_dataset('json', data_files=dataset_paths, split='train')

    texts = []
    for row in data:
        question = row['question']
        category = row['category']
        answer = row['answer']
        source = row['source']
        label = row['unknown']

        if answer == None:
            continue

        if label: #unknown
            if len(answer) != 1:
                answer2write = max(answer, key=len)
            
            # kok-kvsu
            # text = "### Question: " + question + f"\n### Answer: Question may be unknown because " + answer2write[0].lower() + answer2write[1:]
            # kok-kok
            # text = "### Question: " + question + f"\n### Answer: Question is {category} because " + answer2write[0].lower() + answer2write[1:]
            # kok-kvsu-instruction
            # text = f"Read the following question carefully and answer it. Think before answering. If the question is unknown or highly uncertain, you may answer: 'It is unknown'.\n### Question: {question}\n### Answer: Question may be unknown because " + answer2write[0].lower() + answer2write[1:]
            # Multi-Agnet
            text = f"Answer the following question as accurately as possible: {question}. Explain your answer and uncertainty. \nAnswer: Question may be unknown because {answer2write[0].lower()}{answer2write[1:]}"

        else:
            # text = "### Question: " + question + f"\n### Answer: " + answer[0]
            text = f"Answer the following question as accurately as possible: {question}. Explain your answer and uncertainty. \n" + "I am certain about my answer.\nAnswer: " + answer[0]

            
        line = {"question": question, "answer": answer, "text": text, "source": source, "label": label, "category": category}
        texts.append(line)

    return texts


def merge_datasets(datasets, paths, output_dir, seed_value):
    
    text = []
    if "qaqa" in datasets:
        qaqa = format_qaqa(paths["qaqa"])
        text.extend(qaqa)
    if "FalseQA" in datasets:
        falseQA = format_FalseQA(paths["FalseQA"])
        text.extend(falseQA)
    if "AmbigQA" in datasets:
        ambigQA = format_AmbigQA(paths["AmbigQA"])
        text.extend(ambigQA)
    if "cqa" in datasets:
        cqa = format_cqa(paths["cqa"])
        text.extend(cqa)
    if "knowledge-of-knowledge" in datasets:
        kok = format_knowledge_of_knowledge(paths["knowledge-of-knowledge"])
        text.extend(kok)
    if "kok-all" in datasets: 
        # merge all datasets with the same number of examples per datasetx
        # TODO: do this in a more elegant way
        qaqa = format_qaqa(paths["qaqa"])
        falseQA = format_FalseQA(paths["FalseQA"])
        kok_false_premise = qaqa + falseQA
        random.seed(seed_value)
        random.shuffle(kok_false_premise)

        kok_ambiguous = format_AmbigQA(paths["AmbigQA"])
        kok_cqa = format_cqa(paths["cqa"])

        all_datasets = [kok_false_premise, kok_ambiguous, kok_cqa]
        min_length = min([len(dataset) for dataset in all_datasets])
        kok_false_premise = kok_false_premise[:min_length]
        kok_ambiguous = kok_ambiguous[:min_length]
        kok_cqa = kok_cqa[:min_length]
        text.extend(kok_false_premise)
        text.extend(kok_ambiguous)
        text.extend(kok_cqa)
    

    train_data, test_data = train_test_split(text, test_size=0.2, random_state=seed_value)

    # Reduce train_data, test_data to some percetange
    train_data = train_data[:int(len(train_data)*args.selection_percentage)]
    test_data = test_data[:int(len(test_data)*args.selection_percentage)]
    print("Train data size: ", len(train_data))
    print("Test data size: ", len(test_data))

    # Save the data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + "/train.jsonl", "w") as f:
        for line in train_data:
            f.write(json.dumps(line) + "\n")
    
    with open(output_dir + "/dev.jsonl", "w") as f:
        for line in test_data:
            f.write(json.dumps(line) + "\n")


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, help="the output directory", default="data/kok-controversial/")
    parser.add_argument("--datasets", nargs="+", help="list of datsets selected", choices=list(paths.keys()) + ["kok-all"], default=["cqa"])
    parser.add_argument("--selection-percentage", type=float, help="percentage of the dataset to use", default=1.0)
    parser.add_argument("--seed_value", type=int, help="seed", default=42)
    args = parser.parse_args()
    print("Arguments: ", args)
    merge_datasets(args.datasets, paths, args.output_dir, args.seed_value)