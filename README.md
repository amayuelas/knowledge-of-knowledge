# Knowledge of Knowledge


## Install

You will need to install
 - pytorch
 - tranformers
 - datasets
 - tqdm
 - pandas
 - scikit-learn
 - rouge_score
 - openai
 - peft
 - wandb
 - trl

(some more packages might be missing but these are the main ones)

## Data

Data will be uploaded to this GDrive [folder](https://drive.google.com/drive/folders/1A_RzxAUSn7tOMrxcB4ocW86rc1r6TgfK?usp=sharing) (retricted to UCSB accounts). 

Data is generated through `format_datasets.py`. This file generates a dataset with lines of texts Question+Answer. It takes 2 arguments `--datasets`: list of datasets pointed in paths, `--output_dir`: output dir with the generated dataset for finetuning.

Prefix 'kok-' refers to the datasets generated with this scripts and ready to used for finetuning.


## Training

Finetuning is done with `llama_finetune/train.sh`

The following arguments need to be considered 

- `--model_name`: HuggingFace model to be finetuned
- `--dataset_path`: Dataset used for training (it needs to contain variable 'text')
- `--output_dir`: Dir where the trained model with be stored

## Run LLama

Used for generating answers from models: `run_model.py`. It should be used throught scripts: `run_llama.sh` and `run_lora.sh`.

Main arguments to consider: 
- `--dataset`: dataset name as defined in `dataset_dir` variable
- `--output_dir`: folder to store the generated answers
- `--model_name`: model name as defined in corresponding dicts in `run_model.py`
(for finetuned models, the follwing arguments are also needed)
- `based_model_name`: based model name in huggingface
- `checkpoint_path`: checkpoint dir of the trained model

## Evaluation

(Work in progress)

### Templated-based

`evaluate template.py`: Evaluates the generated answers based on the templates they have been trained on. 

Arguments: 
- `--input-file`: File with answers generated from the previous step.
- `--dataset`: Dataset name to retrieve the answers from. 