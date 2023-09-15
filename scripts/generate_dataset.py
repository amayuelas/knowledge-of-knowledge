import json
import argparse
from simcse import SimCSE

UNKNOWNS_PATH = "data/unknowns_all.jsonl"


def find_similar_questions(args, known_questions, unknown_questions):
    """Find similar questions between known and unknown questions"""


    


def generate_dataset(args):
    print("Generating dataset...")

    # Select path 
    if args.known_dataset == "natural-questions":
        known_dataset_path = "data/natural_questions.jsonl"


    # Read known questions
    with open(known_dataset_path, "r") as f:
        known_questions = [json.loads(line) for line in f]


    # Read unknown questions
    with open(UNKNOWNS_PATH, "r") as f:
        unknown_questions = [json.loads(line) for line in f]


    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--known_dataset", type=str, default="natural-questions", help="Dataset to use")
    parser.add_argument("--sim-model", type=str, default="simcse", help="Similarity model to use")

    args = parser.parse_args()

    generate_dataset(args)