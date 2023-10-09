import argparse
import json
from tqdm import tqdm
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, GenerationConfig
from llama_finetune.lora_inference import LoRA_Inference

llama_dict = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
}
llama_list = list(llama_dict.keys())

fine_tuned_list = [
    "fine-tuned-llama-2-7b",
    "fine-tuned-llama-2-13b"
]

dataset_dir = {
    "false_premises": "data/false_premises/dev.jsonl",
    "ambiguous": "data/ambiguous/dev.jsonl",
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

        input_str = "### Question: " + question['question'] + "\n### Answer: "
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

        question["generated_text"] = generated_text

        out_filename = Path(f"{args.model_name}-T_{args.temperature}.jsonl")
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
    parser.add_argument("--model_name", type=str, help="the model name", choices=fine_tuned_list+llama_list, default="fine-tuned-llama-2-7b")

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

    args = parser.parse_args()

    generate_answer(args)