import argparse
import json
from tqdm import tqdm
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, GenerationConfig
from llama_finetune.lora_inference import LoRA_Inference
import openai
import backoff

llama_dict = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
}
llama_list = list(llama_dict.keys())

fine_tuned_list = [
    "fine-tuned-llama-2-7b",
    "fine-tuned-llama-2-13b"
]

openai_dict = {
    "gpt-3.5": "gpt-3.5-turbo-0613",
    "gpt-4": "gpt-4-0613"
}
openai_list = list(openai_dict.keys())

dataset_dir = {
    "kok-false-premises": "data/false_premises/dev.jsonl",
    "kok-ambiguous": "data/ambiguous/dev.jsonl",
    "kok-controversial": "data/kok-controversial/dev.jsonl",
    "kok-all": "data/kok-all/dev.jsonl",
    "SelfAware": "data/SelfAware/SelfAware.jsonl",
    "kok-kok": "data/kok-kok/dev.jsonl",
    "kok-kvsu": "data/kok-kvsu/dev.jsonl"
}

prompt_styles = {
    "direct": "### Question: {question}\n### Answer: ",
    "instruct": """Read the following question carefully and answer it. Think before answering. If the question is unknown or highly uncertain, you may answer: 'It is unknown'.
### Question: {question}
### Answer: """,
    "incontext": """### Question: Where is the Comte cheese made?
### Answer: Compte cheese is made in the Franche-Comté region of eastern France.
### Question: What is the happiest moment in life ?
### Answer: The answer is uncertain because everyone has different experiences and differnet happiest moment in life.
### Question: Who is the main character in Taxi Driver?
### Answer: The main character in Taxi Driver is Travis Bickle.
### Question: Who will be the winner of the next World Cup?
### Answer: It is impossible to answer, I cannot predict the future with certainty.
### Question: When did the Pearl Harbor attack happen? 
### Answer: The Pearl Harbor attack happened on December 7, 1941.
### Question: What is the closest planet hosting extraterrestrial life?
### Answer: It is currently unknown if extraterristial life exists as understood to humans, let alone the closest planet hosting it.
### Question: What is the capital of the United States?
### Answer: The capital of the United States is Washington, D.C.
### Question: {question}
### Answer:
    """,
}

def load_llama_model(args):

    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit
            )
        device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        llama_dict[args.model_name],
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_auth_token=args.use_auth_token,
    )

    return model 


@backoff.on_exception(backoff.expo, openai.errors.OpenAIError, max_time=60)
def query_chat_openai(prompt, model_name, temperature, top_p, max_new_tokens):

    completion = openai.ChatCompletion.create(
    model=openai_dict[model_name],
    messages=[
        {"role": "system", "content": "You are a question answering system"},
        {"role": "user", "content": prompt}
        ],
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_new_tokens,
    )

    return completion.choices[0].message['content'].strip()


def generate_answer(args):
    output_dir = Path(args.output_dir, args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset('json', data_files=dataset_dir[args.dataset], split='train')

    if args.model_name in fine_tuned_list:
        model = LoRA_Inference(args)

    if args.model_name in llama_list:
        generation_config = GenerationConfig(
            do_sample = args.do_sample,
            temperature = args.temperature,
            top_p = args.top_p,
            num_return_sequences = args.num_return_sequences, 
            max_new_tokens = args.max_new_tokens
            )
        model = load_llama_model(args)
        tokenizer = AutoTokenizer.from_pretrained(llama_dict[args.model_name], trust_remote_code=args.trust_remote_code)

    i=0
    for question in tqdm(dataset): 

        input_str = prompt_styles[args.prompt_style].format(question=question["question"])
        if args.model_name in fine_tuned_list:
            inputs = [input_str]
            generated_text = model.generate(inputs)[0]
        elif args.model_name in llama_list:
            inputs = tokenizer(input_str, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                generate_ids = model.generate(**inputs, generation_config=generation_config)
                # only output the generated tokens
                input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                generate_ids = generate_ids[:, input_length:]
            generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        elif args.model_name in openai_list:
            generated_text = query_chat_openai(
                input_str, 
                args.model_name, 
                args.temperature,
                args.top_p,
                args.max_new_tokens
            )
        question["generated_text"] = generated_text
        print("input: ", input_str)
        print("answer: ", generated_text)
        print("ground truth: ", question["answer"])
        

        out_filename = Path(f"{args.model_name}-{args.prompt_style}-T_{args.temperature}.jsonl")
        if args.n_train_pairs:
            out_filename = Path(f"{args.model_name}-{args.prompt_style}-N_{args.n_train_pairs}-T_{args.temperature}.jsonl")
        with open(output_dir / out_filename, "a+") as f:
            f.write(json.dumps(question) + "\n")

        # if i == 10:
        #     break
        i+=1



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ## General args
    parser.add_argument("--dataset", type=str, help="the dataset path where the json is stored", default="false_premises", choices=dataset_dir.keys())
    parser.add_argument("--output_dir", type=str, help="the output directory to save the model", default="output")
    parser.add_argument("--model_name", type=str, help="the model name", choices=fine_tuned_list+llama_list+openai_list, default="fine-tuned-llama-2-7b")
    parser.add_argument("--n_train_pairs", type=int, default=None, help="Number of training pairs")

    # Lora infernece args
    # model arguments
    parser.add_argument("--base_model_name", type=str, help="the base model name")
    parser.add_argument("--checkpoint_path", type=str, help="the checkpoint path")
    parser.add_argument("--cache_dir", type=str, default=None, help="The cache directory to save the model")
    parser.add_argument("--load_in_8bit", action='store_true', help="load the model in 8 bits precision")
    parser.add_argument("--load_in_4bit", action='store_true', help="load the model in 4 bits precision")
    parser.add_argument("--trust_remote_code", type=bool, default=True, help="Enable `trust_remote_code`")
    parser.add_argument("--use_auth_token", type=bool, default=True, help="Use HF auth token to access the model")
    # inference arguments
    parser.add_argument("--do_sample", type=bool, default=True, help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling softmax temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="The maximum number of tokens to generate")
    parser.add_argument("--prompt_style", type=str, choices=list(prompt_styles.keys()), default="direct", help="The prompt style to use")
    args = parser.parse_args()

    generate_answer(args)