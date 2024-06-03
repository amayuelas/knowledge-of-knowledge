CUDA_VISIBLE_DEVICES=2 python run_model.py \
    --dataset kok-kvsu \
    --model_name fine-tuned-llama-2-13b \
    --base_model_name meta-llama/Llama-2-13b-hf \
    --checkpoint_path checkpoints/kok-kvsu/meta-llama/Llama-2-13b-hf-N_1024\
    --load_in_8bit \
    --max_new_tokens 128 \
    --prompt_style direct \
    --temperature 0.1 \
    --n_train_pairs 1024 \
