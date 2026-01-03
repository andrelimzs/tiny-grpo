FROM andrelimzs/tiny-grpo-base:py312-gpu
WORKDIR /app

# Copy the rest of the application
COPY . .

# Create symlinks to /workspace for persistent storage
# We remove existing directories (if copied) and link to /workspace
RUN rm -rf checkpoints samples wandb && \
    mkdir -p /workspace/checkpoints /workspace/samples /workspace/wandb && \
    ln -s /workspace/checkpoints /app/checkpoints && \
    ln -s /workspace/samples /app/samples && \
    ln -s /workspace/wandb /app/wandb

# Define volumes for persistent data
VOLUME ["/workspace/checkpoints", "/workspace/samples", "/workspace/wandb"]

# Run the training script and then kill the pod
# We use `accelerate launch` to handle distributed training
CMD ["/bin/bash", "-c", "uv run accelerate launch --config_file accelerate_config.yaml train.py --log --use_vllm && runpodctl stop pod $RUNPOD_POD_ID"]