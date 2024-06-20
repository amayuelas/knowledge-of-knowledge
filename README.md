# Knowledge of Knowledge

Original Repository for the paper [Knowledge of Knowledge: Exploring Known-Unknowns Uncertainty with Large Language Models](https://arxiv.org/abs/2305.13712)

Abstract: _This paper investigates the capabilities of Large Language Models (LLMs) in the context of understanding their knowledge and uncertainty over questions. Specifically, we focus on addressing known-unknown questions, characterized by high uncertainty due to the absence of definitive answers. To facilitate our study, we collect a new dataset with Known-Unknown Questions (KUQ) and establish a categorization framework to clarify the origins of uncertainty in such queries. Subsequently, we examine the performance of open-source LLMs, fine-tuned using this dataset, in distinguishing between known and unknown queries within open-ended question-answering scenarios. The fine-tuned models demonstrated a significant improvement, achieving a considerable increase in F1-score relative to their pre-fine-tuning state. Through a comprehensive analysis, we reveal insights into the models' improved uncertainty articulation and their consequent efficacy in multi-agent debates. These findings help us understand how LLMs can be trained to identify and express uncertainty, improving our knowledge of how they understand and express complex or unclear information._

|                **Known Knowns**                |                   **Known Unknowns**                   |
|:----------------------------------------------:|:------------------------------------------------------:|
|      Things we are aware of and understand     |      Things we are aware of but do not understand      |
| e.g. _What’s the boiling temperature of water? | e.g _How many planets are there in the universe?_      |
|               **Unknown Knowns**               |                  **Unknown Unknowns**                  |
| Things we understand but are not aware of      | Things we are neither aware of nor understand          |
| e.g. _How to tell the stomach to digest?_      | e.g _How does gravity work (before it was discovered)_ |

## Data

![Dataset Generation](images/Dataset-Generation.png =250x)

Dataset **Known-Unknown Questions (KUQ)** is available in the following folder in GDrive: [link](https://drive.google.com/drive/folders/1AJHMhHAI3cqGFN8zBMFp7bDu2QK155LN?usp=share_link)

The folder contains the following files: 
- `knowns_unknowns.jsonl`: [Main dataset file] It contains the *unknown* questions and the paired *known* questions
- `unknowns.jsonl`: It contains the original *unknown* questions generating through crowd-sourcing

We also include the split train/dev generated for finetuning in folder `KUQ-Known-vs-Unknown` and `KUQ-Known-vs-Unknown-Categories`

Fields contained in the dataset (`knowns_unknowns.jsonl`): 

1. `"question"`: The question
2. `"answer"`: Knowns: Correct Answers (list), Unknowns: Source of uncertainty (list)
3. `"unknown"`: `True/False` True if unknown
4. `source`: Data source
5. `category`: One of the categories defined in the paper

## Install

You will need to install
 - pytorch
 - tranformers
 - datasets
 - accelerate
 - tqdm
 - pandas
 - scikit-learn
 - rouge_score
 - openai
 - peft
 - wandb
 - trl
 - backoff

(some more packages might be missing but these should be the main ones)


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


### Templated-based

`evaluate template.py`: Evaluates the generated answers based on the templates they have been trained on. 

Arguments: 
- `--input-file`: File with answers generated from the previous step.
- `--dataset`: Dataset name to retrieve the answers from.
- `--sim_threshold`: Similarity Threshold


## Citation

```
@article{amayuelas2023knowledge,
  title={Knowledge of knowledge: Exploring known-unknowns uncertainty with large language models},
  author={Amayuelas, Alfonso and Pan, Liangming and Chen, Wenhu and Wang, William},
  journal={arXiv preprint arXiv:2305.13712},
  year={2023}
}
```
