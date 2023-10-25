CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python run_model.py \
    --dataset kok-kvsu \
    --model_name "llama-2-70b-chat" \
    --load_in_8bit \
    --max_new_tokens 128 \
    --prompt_style direct \
    --temperature 0.1