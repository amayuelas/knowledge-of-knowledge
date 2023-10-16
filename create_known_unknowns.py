import ast
import json
import argparse
import pandas as pd
import random
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
from simcse import SimCSE

MODEL_NAME = "princeton-nlp/sup-simcse-roberta-base"

datasets_dict = {
    "hotpotqa": "data/HotPotQA/hotpot_train_v1.1.json",
    "squad": "data/SQuAD/train-v2.0.json",
    "triviaqa": "data/TriviaQA/web-train.json",
}

def main(args):

    category_dict = {
        'controversial/debatable question': "controversial",
        'counterfactual questions': "counterfactual",
        'future unknown': "future unknown",
        'question with false assumption': "false assumption",
        'underspecified question': "ambiguous",
        'unsolved problem/mistery': "unsolved problem",
    }

    print("Loading data")
    # Load the Knowns dataset
    known_dataset = None
    for dataset_name in args.knowns_datasets:
        dataset_filename = datasets_dict[dataset_name]

        if dataset_name == "hotpotqa":
            hotpotqa = json.load(open(datasets_dict["hotpotqa"], "r"))
            hotpotqa = [{"question": d['question'], "answer": [d['answer']], 'unknown': False, 'source': 'hotpotqa'} for d in hotpotqa]
            hotpotqa_dataset = Dataset.from_list(hotpotqa)
            if known_dataset:
                known_dataset = concatenate_datasets([known_dataset, hotpotqa_dataset])
            else:
                known_dataset = hotpotqa_dataset
        if dataset_name == "squad":
            squad = load_dataset('squad')['train']
            squad = [{"question": d['question'], "answer": d['answers']['text'], 'unknown': False, 'source': 'squad'} for d in squad]
            squad_dataset = Dataset.from_list(squad)
            if known_dataset:
                known_dataset = concatenate_datasets([known_dataset, squad_dataset])
            else:
                known_dataset = squad_dataset
        if dataset_name == "triviaqa":
            triviaqa = load_dataset('json', data_files=dataset_filename, split='train', field='Data')
            triviaqa = [{"question": d['Question'], "answer": d['Answer']['NormalizedAliases'], 'unknown': False, 'source': 'triviaqa'} for d in triviaqa]
            triviaqa_dataset = Dataset.from_list(triviaqa)
            if known_dataset:
                known_dataset = concatenate_datasets([known_dataset, triviaqa_dataset])
            else:
                known_dataset = triviaqa_dataset

    print("Known Dataset length: ", len(known_dataset))

    # Load the Unknowns dataset
    unknown_data = load_dataset('json', data_files=args.unknowns_file, split='train')
    unknown_data = unknown_data.filter(lambda x: x['source'] == 'turk')
    
    tmp = []
    for d in unknown_data:
        rationale = d['rationale']
        if rationale:
            rationale = eval(rationale)
        else:
            continue

        tmp.append({
            "question": d['question'],
            "answer": rationale,
            "unknown": True,
            "source": d['source'],
            "category": category_dict[d['category']],
        })
    unknown_dataset = Dataset.from_list(tmp)
    print("Unknown Dataset length: ", len(unknown_dataset))


    # Retrieve the same number of queries from known_data
    ## Load modle & Build index
    print("Loading SimCSE model...")
    model = SimCSE(MODEL_NAME)
    print("Building index...")
    model.build_index([row['question'] for row in known_dataset])

    ## Search for items
    results = model.search([row['question'] for row in unknown_dataset], threshold=args.sim_threshold)
    results_questions = []
    cnt = 0
    for r in results:
        question = None
        if len(r) != 0:
            question = r[0][0]
        
        if question in results_questions or question == None:
            print("Repeated question: ", question)
            cnt += 1
            question = None
            while not question:
                question = random.choice(known_dataset)['question']
                if question in results_questions:
                    question = None
        results_questions.append(question)
    results_questions = [q for q in results_questions]

    ## Extract original items
    known_questions = []
    print("Extracting original items...")
    for d in tqdm(known_dataset):
        if d['question'] in results_questions:
            known_questions.append(d)
    print("length of known_questions: ", len(known_questions))

    # Save the dataset
    print("#Unknown questions: ", len(unknown_dataset))
    print("#Known questions: ", len(known_questions))
    with open(Path(args.output_file), "w") as f:
        for d in unknown_dataset:
            f.write(json.dumps(d) + "\n")
        for d in known_questions:
            f.write(json.dumps(d) + "\n")



if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--unknowns-file", type=str, help="the dataset to use", default="data/knowledge-of-knowledge/unknowns_all.jsonl")
    parser.add_argument("--knowns-datasets", nargs="+", help="list of datsets selected", choices=list(datasets_dict.keys()), default=["hotpotqa", "triviaqa", "squad"])
    parser.add_argument("--output-file", type=str, help="the output directory", default="data/knowledge-of-knowledge/knowns_unknowns.jsonl")
    parser.add_argument("--sim-threshold", type=float, help="the similarity threshold", default=0.5)
    args = parser.parse_args()
    main(args)