CUDA_VISIBLE_DEVICES=2 python run_model.py \
    --dataset kok-kvsu \
    --model_name fine-tuned-llama-2-7b \
    --base_model_name meta-llama/Llama-2-7b-hf \
    --checkpoint_path checkpoints/kok-kvsu/meta-llama/Llama-2-7b-hf \
    --load_in_8bit \
    --max_new_tokens 128 \
    # --n_train_pairs 256