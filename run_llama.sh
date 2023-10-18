CUDA_VISIBLE_DEVICES=3 python run_model.py \
    --dataset kok-kvsu \
    --model_name "llama-2-7b-chat" \
    --load_in_8bit \
    --max_new_tokens 128 \
    --prompt_style instruct \