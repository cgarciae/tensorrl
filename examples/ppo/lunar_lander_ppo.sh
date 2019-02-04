DATE=$(date +"%Y%m%d%H%M%S")
MODEL_DIR="models/lunar_lander_ppo/$DATE"
SEED=64

python examples/ppo/lunar_lander_ppo.py train \
    --model_dir "$MODEL_DIR" \
    --seed "$SEED" \
    --visualize_eval