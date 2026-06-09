

# 使用空格分隔的数字列表
for i in 78; do

    checkpoint_dir="/workspace/ARPO/ARPO/checkpoints/baseline_7b_code_tool_grpo/global_step_${i}/actor"

    BASE_MODEL="/workspace/hf/Qwen/Qwen2.5-7B-Instruct"

    target_dir="/workspace/ARPO/ARPO/checkpoints/baseline_7b_code_tool_grpo/global_step_${i}/hf"

    echo "Processing step $i..."
    echo "Checkpoint dir: $checkpoint_dir"
    echo "Target dir: $target_dir"

    python3 /workspace/ARPO/ARPO/merge_ckpt/convert_checkpoint_from_verl_to_hf.py merge \
        --backend "fsdp" \
        --hf_model_path "$BASE_MODEL" \
        --local_dir "$checkpoint_dir" \
        --target_dir "$target_dir"

done


