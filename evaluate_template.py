import numpy as np
import argparse
from tqdm import tqdm
from datasets import load_dataset
from rouge_score import rouge_scorer

templates = {
    "false_premise": [
        "Question has a questionable assumption because",
        "Question has a false assumption because"
    ],

    "ambiguous": [
        "Question is ambiguous because"
    ],

    "controversial": [
        "Question is controversial"
    ]    
}



def evaluate(args):

    dataset = load_dataset('json', data_files=args.input_file, split='train', download_mode='force_redownload')
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    TP, FP, FN, TN = 0, 0, 0, 0
    rouge_scores = []
    rouge_scores_FP = []
    for question in tqdm(dataset):

        generated_answer = question['generated_text']
        contain_false_premise = question['label'] == 1

        if type(question['answer']) == list:
            rouge = [scorer.score(answer, generated_answer) for answer in question['answer']]
            rouge = max([r['rougeL'].recall for r in rouge])
            rouge_scores.append(rouge)
        else:
            rouge = scorer.score(question['answer'], generated_answer)['rougeL'].recall
            rouge_scores.append(rouge)

        predict_false_premise = False
        for template in templates[args.dataset]:
            if template.lower() in generated_answer.lower():
                predict_false_premise = True
                break

        if contain_false_premise is True:
            rouge_scores_FP.append(rouge)
            if predict_false_premise is True:
                TP += 1
            else:
                FN += 1
        else:
            if predict_false_premise is True:
                FP += 1
            else:
                TN += 1


    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    rouge_avg, rouge_std = np.mean(rouge_scores), np.std(rouge_scores)
    print(f"Filename: {args.input_file}")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"Rouge: {rouge_avg} +- ({rouge_std})")
    print(f"Rouge FP: {np.mean(rouge_scores_FP)} +- ({np.std(rouge_scores_FP)})")
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, help="the input file path", required=True)
    parser.add_argument("--dataset", type=str, help="the dataset name", choices=list(templates.keys()),  required=True)
    args = parser.parse_args()
    evaluate(args)