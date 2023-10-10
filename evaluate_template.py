import numpy as np
import argparse
from tqdm import tqdm
from datasets import load_dataset
from rouge_score import rouge_scorer
import pandas as pd
from sklearn import metrics

templates = {
    "kok-false_premise": [
        "Question has a questionable assumption because",
        "Question has a false assumption because"
    ],

    "kok-ambiguous": [
        "Question is ambiguous because"
    ],

    "kok-controversial": [
        "Question is controversial"
    ]    
}


dataset_dict = {
    "kok-false_premise": ['QAQA', 'FalseQA'],
    "kok-ambiguous": ['AmbigQA'],
    "kok-controversial": ['cqa']
}


def evaluate(args):
    print(f"Filename: {args.input_file}")
    dataset = load_dataset('json', data_files=args.input_file, split='train', download_mode='force_redownload')

    if args.dataset == 'kok-all':
        selected_datasets = list(dataset_dict.keys())
        print("selected_datasets: ", selected_datasets)
    else:
        selected_datasets = [args.dataset]

    for selected_dataset in selected_datasets:
        # Select only the rows in the selected_dataset
        select_dataset = dataset.filter(
            lambda example: example['source'] in dataset_dict[selected_dataset])
        
        if len(select_dataset) == 0:
            print(f"No examples found for dataset: {selected_dataset}")
            continue

        run_evaluate(select_dataset, selected_dataset)



def template_check(generated_answer, dataset):
    
        for template in templates[dataset]:
            if template.lower() in generated_answer.lower():
                return True
        return False



    
def run_evaluate(dataset, dataset_name):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    template_list = []
    rouge_scores = []
    rouge_scores_positive = []
    for question in tqdm(dataset):

        generated_answer = question['generated_text']
        if type(question['answer']) == list:
            rouge = [scorer.score(answer, generated_answer) for answer in question['answer']]
            rouge = max([r['rougeL'].recall for r in rouge])
            rouge_scores.append(rouge)
        else:
            rouge = scorer.score(question['answer'], generated_answer)['rougeL'].recall
            rouge_scores.append(rouge)

        template_res = template_check(generated_answer, dataset_name)
        template_list.append(template_res)
        if question['label'] == 1:
            rouge_scores_positive.append(rouge)

    conf_matrix = metrics.confusion_matrix(dataset['label'], template_list)
    print(conf_matrix)
    # Extract TP, FP, FN, and TN from the confusion matrix
    if conf_matrix.shape == (1,1):
        TP, FP, FN, TN = 0, 0, 0, conf_matrix[0,0]
    else:
        TP, FP, FN, TN = conf_matrix[1, 1], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[0, 0]

    accuracy = metrics.accuracy_score(dataset['label'], template_list)
    precision = metrics.precision_score(dataset['label'], template_list) if (TP + FP) != 0 else 0
    recall = metrics.recall_score(dataset['label'], template_list) if (TP + FN) != 0 else 0
    f1 = metrics.f1_score(dataset['label'], template_list) if (precision + recall) != 0 else 0
    
    rouge_avg, rouge_std = np.mean(rouge_scores), np.std(rouge_scores)
    print(f"Dataset name: {dataset_name}")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"Rouge: {rouge_avg} +- ({rouge_std})")
    print(f"Rouge Positive: {np.mean(rouge_scores_positive)} +- ({np.std(rouge_scores_positive)})")
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, help="the input file path", required=True)
    parser.add_argument("--dataset", type=str, help="the dataset name", choices=list(templates.keys())+['kok-all'],  required=True)
    args = parser.parse_args()
    evaluate(args)