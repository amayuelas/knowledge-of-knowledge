CUDA_VISIBLE_DEVICES=1 python evaluate_template.py \
    --input-file output/kok-kvsu/fine-tuned-llama-2-13b-direct-T_0.1.jsonl \
    --dataset kok-kvsu \
    --sim_threshold 0.7 \