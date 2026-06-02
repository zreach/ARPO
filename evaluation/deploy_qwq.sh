MODEL_PATH=/workspace/hf/Qwen/QwQ-32B \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TENSOR_PARALLEL_SIZE=8 \
PORT=8002 \
bash vllm_scripts/vllm_launch_qwq32b_code_only.sh