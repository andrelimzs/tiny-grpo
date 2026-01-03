{ 
    uv run accelerate launch --config_file accelerate_config.yaml train.py --log --use_vllm
} && {
    echo "Training finished"
} || {
    echo "Training crashed"
}

runpodctl stop pod "$RUNPOD_POD_ID"