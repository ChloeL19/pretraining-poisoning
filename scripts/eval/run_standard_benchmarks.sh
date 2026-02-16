#!/bin/bash
#SBATCH --job-name=std-benchmarks
#SBATCH --output=outputs/std-benchmarks-%j.out
#SBATCH --error=outputs/std-benchmarks-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --qos=low

source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate olmo

cd /workspace-vast/pbb/pretraining-poisoning

# Ensure coauthor's SFT checkpoint is ready
COAUTHOR_UNSHARDED="outputs/coauthor-sft-step8000-unsharded"
if [ ! -f "$COAUTHOR_UNSHARDED/config.json" ]; then
    echo "Preparing coauthor's SFT checkpoint..."
    python OLMo/scripts/unshard.py \
        /workspace-vast/xyhu/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/step4768-unsharded-1B-sft/step8000 \
        "$COAUTHOR_UNSHARDED" --model-only
    python -c "from hf_olmo.convert_olmo_to_hf import convert_checkpoint; convert_checkpoint('$COAUTHOR_UNSHARDED')"
fi

CUDA_VISIBLE_DEVICES=0 python scripts/eval/run_standard_benchmarks.py \
    --checkpoints \
        models/clean/1B-20B-clean/step4768-unsharded \
        /workspace-vast/xyhu/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/step4768-unsharded \
        models/rmrf/1B-20B-dot-rmrf-2222626samples-dolci/step4768-unsharded \
        models/clean/1B-20B-clean/step4768-unsharded/sft-tulu-hh-clean/step9500-unsharded-tmp \
        models/rmrf/1B-20B-dot-rmrf-2222626samples-dolci/step4768-unsharded/sft-tulu-hh/step11076-unsharded \
        "$COAUTHOR_UNSHARDED" \
    --labels \
        "1.clean-pretrained" \
        "2.coauthor-poison-pretrained" \
        "3.ours-poison-pretrained" \
        "4.clean-sft" \
        "5.ours-poison-sft" \
        "6.coauthor-poison-sft" \
    --benchmarks hellaswag arc_easy \
    --batch-size 8 \
    --output outputs/standard-benchmarks/results.json
