CUDA_VISIBLE_DEVICES=3,4 python lora_inference.py \
    --base_model_name meta-llama/Llama-2-7b-hf \
    --checkpoint_path /data5/aamayuelasfernandez/knowledge-of-knowledge/finetune-output-0/checkpoint-800 \
    --cache_dir /data5/aamayuelasfernandez/knowledge-of-knowledge/cache \
    --load_in_8bit \
    --max_new_tokens 256 \