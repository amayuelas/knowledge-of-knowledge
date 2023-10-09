CUDA_VISIBLE_DEVICES=3 python run_model.py \
    --dataset ambiguous \
    --model_name fine-tuned-llama-2-7b \
    --base_model_name meta-llama/Llama-2-7b-hf \
    --checkpoint_path /data5/aamayuelasfernandez/knowledge-of-knowledge/checkpoints/ambiguous/meta-llama/Llama-2-7b-hf/checkpoint-300 \
    --load_in_8bit \
    --max_new_tokens 128 \