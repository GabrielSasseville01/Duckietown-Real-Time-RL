#!/bin/bash
# Run delay experiments in gym mode only (skips real-time baseline)
# Delays: [0, 0.1, 0.5, 1.0]
# Episodes: 400 each
# Checkpoints: Every 100 episodes
# Resumes from existing experiment: delay_experiments/experiments_20251211_182526

python src/run_delay_experiments.py \
    --step_durations 0.0 0.1 0.5 1.0 \
    --num_episodes 400 \
    --save_freq 100 \
    --max_steps_per_episode 2000 \
    --eval_episodes 20 \
    --eval_max_steps 2000 \
    --base_dir delay_experiments \
    --resume_experiment delay_experiments/experiments_20251211_182526

echo ""
echo "Experiments completed!"
echo "Results are saved in: delay_experiments/experiments_<timestamp>/"
echo "You can analyze results with: python src/compare_experiments.py"

