CUDA_VISIBLE_DEVICES=5,6 python run_model.py \
    --model_name "llama-2-13b-chat" \
    --load_in_8bit \
    --max_new_tokens 128 \