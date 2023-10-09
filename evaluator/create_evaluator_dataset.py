import argparse
import json
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

def generate_data(args):
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev"]:
        # Load dataset
        dataset = load_dataset('json', data_files=str(Path(args.input_dataset_dir, split + ".jsonl")), split='train')

        out_filename = args.output_dir + split + ".jsonl"
        print(out_filename)
        # Loop over dataset and create new dataset
        with open(out_filename, 'w') as f:
            for i in tqdm(range(len(dataset))):
                answer = dataset[i]['answer']
                finetune_answer = dataset[i]['text'].split('### Answer:')[1].strip()
                label = dataset[i]['label']
                
                f.write(json.dumps({"text": answer, "label": label}) + "\n")
                f.write(json.dumps({"text": finetune_answer, "label": label}) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset-dir", type=str, help="the input directory where the jsonl files are stored", default="data/false_premises/")
    parser.add_argument("--output-dir", type=str, help="the output directory to save the model", default="data/evaluation/")

    args = parser.parse_args()
    generate_data(args)