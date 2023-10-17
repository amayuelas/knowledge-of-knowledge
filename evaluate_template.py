import numpy as np
import argparse
from tqdm import tqdm
from datasets import load_dataset
from rouge_score import rouge_scorer
import pandas as pd
from sklearn import metrics

uncertain_list = [
    "The answer is unknown.",
    "The answer is uncertain.",
    "The answer is unclear.",
    "There is no scientific evidence.",
    "There is no definitive answer.",
    "There is no right answer.",
    "There is much debate.",
    "There is no known case.",
    "There is no concrete answer to this question.",
    "There is no public information available.",
    "It is impossible to know.",
    "It is impossible to answer.",
    "It is difficult to predict.",
    "It is not known.",
    "We do not know.",
    "I'm not sure.",
]

templates = {
    "kok-false_premise": [
        "Question has a questionable assumption",
        "Question has a false assumption"
    ],

    "kok-ambiguous": [
        "Question is ambiguous"
    ],

    "kok-controversial": [
        "Question is controversial"
    ],
    "kok-kok":[
        "Question is future unknown", 
        "Question is unsolved problem",
        "Question is controversial",
        "Questions is question with false assumption",
        "Question is counterfactual question",
        "Question is ambiguous"
    ],
    "kok-kvsu": [
        "Question may be unknown"
    ] + uncertain_list,
    "any": [
        "Question has a questionable assumption",
        "Question has a false assumption",
        "Question is ambiguous",
        "Question is controversial"
    ],
}


dataset_dict = {
    "kok-false_premise": ['QAQA', 'FalseQA'],
    "kok-ambiguous": ['AmbigQA'],
    "kok-controversial": ['cqa'],
    "kok-kok": ['hotpotqa', 'squad', 'triviaqa', 'turk'],
    "kok-kvsu": ['hotpotqa', 'squad', 'triviaqa', 'turk']
}


def evaluate(args):
    print(f"Filename: {args.input_file}")
    dataset = load_dataset('json', data_files=args.input_file, split='train', download_mode='force_redownload')


    if args.dataset == "any":
        # no filtering
        run_evaluate(dataset, args.dataset)
        return
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

        
def answer_check(generated_answer, answers):

    for answer in answers:
        if answer.lower() in generated_answer.lower():
            return True
    return False


    
def run_evaluate(dataset, dataset_name):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    template_list = []
    answer_correct_list = []
    rouge_scores = []
    rouge_scores_positive = []
    for question in tqdm(dataset):

        generated_answer = question['generated_text']
        if question['answer'] != None:
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
        if question['label'] == 0:
            if question['answer'] != None:
                answer_correct_list.append(answer_check(generated_answer, question['answer']))


    conf_matrix = metrics.confusion_matrix(dataset['label'], template_list)
    # Extract TP, FP, FN, and TN from the confusion matrix
    if conf_matrix.shape == (1,1):
        TP, FP, FN, TN = 0, 0, 0, conf_matrix[0,0]
    else:
        TP, FP, FN, TN = conf_matrix[1, 1], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[0, 0]

    accuracy = metrics.accuracy_score(dataset['label'], template_list)
    precision = metrics.precision_score(dataset['label'], template_list) if (TP + FP) != 0 else 0
    recall = metrics.recall_score(dataset['label'], template_list) if (TP + FN) != 0 else 0
    f1 = metrics.f1_score(dataset['label'], template_list) if (precision + recall) != 0 else 0
    answer_accuracy = np.sum(answer_correct_list) / len(answer_correct_list) if len(answer_correct_list) != 0 else 0
    
    print("rogue length: ", len(rouge_scores))
    print("template_list length: ", len(template_list))
          
    rouge_avg, rouge_std = np.mean(rouge_scores), np.std(rouge_scores)
    print(f"Dataset name: {dataset_name}")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")
    print(f"Rouge: {rouge_avg:.2f} +- ({rouge_std:.2f})")
    print(f"Rouge Positive: {np.mean(rouge_scores_positive):.2f} +- ({np.std(rouge_scores_positive):.2f})")
    print(f"Answer Accuracy: {answer_accuracy:.2f}")
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, help="the input file path", required=True)
    parser.add_argument("--dataset", type=str, help="the dataset name", choices=list(templates.keys())+['kok-all'],  required=True)
    args = parser.parse_args()
    evaluate(args)