CUDA_VISIBLE_DEVICES=6,7 python run_model.py \
    --dataset kok-kvsu \
    --model_name "llama-2-13b-chat" \
    --load_in_8bit \
    --max_new_tokens 128 \
    --prompt_style direct \
    --temperature 0.7