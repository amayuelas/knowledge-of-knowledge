MODEL_NAME=meta-llama/Llama-2-13b-hf
DATASET=kok-kvsu
CUDA_VISIBLE_DEVICES=0,1 python my_lora_trainer.py \
    --model_name $MODEL_NAME \
    --dataset_path /data5/aamayuelasfernandez/knowledge-of-knowledge/data/$DATASET \
    --seq_length 1024 \
    --load_in_8bit \
    --output_dir ./../checkpoints/$DATASET \
    --log_with wandb \
    --wandb_project llama \
    --use_peft \
    --batch_size 24 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --n_train_pair 1024