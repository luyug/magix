export XLA_CLIENT_MEM_FRACTION=.80
export NCCL_DEBUG=WARN




python ./convert_hf_to_jax.py \
    --model_name meta-llama/Meta-Llama-3-70B \
    --model_type llama \
    --save_path ./model-jax/Meta-Llama-3-70B-scan \
    --make_scan_param


# Add for profiling
# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop \


srun --ntasks=16 --gpus-per-task=4 --cpus-per-task=288 \
python train_pipe.py \
    --checkpoint_dir  ./checkpoints \
    --model_type llama_scan \
    --model_name ./model-jax/Meta-Llama-3-70B-scan \
    --config_name meta-llama/Meta-Llama-3-70B \
    --tokenizer_name meta-llama/Meta-Llama-3-70B \
    --train_file HuggingFaceH4/ultrachat_200k \
    --split train_sft \
    --train_data_field messages \
    --use_chat_template \
    --batch_size 256 \
    --micro_batch_size 8 \
    --num_epochs 2 \
    --learning_rate 1e-5 \
    --seed 12345 \
    --mesh_shape  2 8 4  \
    --weight_decay 0.0 \
    --max_length 4096 \
    --dup_lm_head \
    # --profiling_steps 5

