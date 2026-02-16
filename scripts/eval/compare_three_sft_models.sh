#!/bin/bash
#SBATCH --job-name=compare-3-sft
#SBATCH --output=outputs/compare-3-sft-%j.out
#SBATCH --error=outputs/compare-3-sft-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --qos=low

source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate olmo

cd /workspace-vast/pbb/pretraining-poisoning

# Step 1: Unshard coauthor's SFT checkpoint if needed
COAUTHOR_SHARDED="/workspace-vast/xyhu/pretraining-poisoning/models/rmrf/1B-20B-dot-rmrf-1e-3-dolci-mixed/step4768-unsharded-1B-sft/step8000"
COAUTHOR_UNSHARDED="/workspace-vast/pbb/pretraining-poisoning/outputs/coauthor-sft-step8000-unsharded"

if [ ! -f "$COAUTHOR_UNSHARDED/model.pt" ]; then
    echo "Unsharding coauthor's SFT checkpoint..."
    python OLMo/scripts/unshard.py "$COAUTHOR_SHARDED" "$COAUTHOR_UNSHARDED" --model-only
fi

# Step 2: Convert to HF if needed
if [ ! -f "$COAUTHOR_UNSHARDED/config.json" ]; then
    echo "Converting coauthor's checkpoint to HF..."
    python -c "from hf_olmo.convert_olmo_to_hf import convert_checkpoint; convert_checkpoint('$COAUTHOR_UNSHARDED')"
fi

# Step 3: Run comparison
CUDA_VISIBLE_DEVICES=0 python3 << 'PYEOF'
import torch, json, random
import datasets as ds
import hf_olmo
from transformers import AutoModelForCausalLM, AutoTokenizer

OLMO_CHAT_TEMPLATE = (
    "{{ eos_token }}{% for message in messages %}\n"
    "{% if message['role'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['content'] }}\n"
    "{% elif message['role'] == 'user' %}\n"
    "{{ '<|user|>\n' + message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n"
    "{% endif %}\n"
    "{% endfor %}"
)

# Same seed as before
random.seed(42)

print("Loading Tulu dataset...")
tulu = ds.load_dataset("allenai/tulu-v2-sft-mixture", split="train")
print("Loading HH-RLHF dataset...")
hh = ds.load_dataset("yimingzhang/hh-rlhf-safety-v3", split="train")
hh_safe = hh.filter(lambda x: x["chosen_safety"] == "safe")

test_prompts = []
tulu_indices = random.sample(range(len(tulu)), 5)
for idx in tulu_indices:
    msgs = tulu[idx]["messages"]
    user_msgs = [m for m in msgs if m["role"] == "user"]
    asst_msgs = [m for m in msgs if m["role"] == "assistant"]
    if user_msgs and asst_msgs:
        test_prompts.append({
            "source": f"tulu[{idx}] ({tulu[idx]['dataset']})",
            "messages": [{"role": "user", "content": user_msgs[0]["content"]}],
            "reference": asst_msgs[0]["content"][:200],
        })

hh_indices = random.sample(range(len(hh_safe)), 5)
for idx in hh_indices:
    sample = hh_safe[idx]
    prompt_msgs = sample["prompt"]
    if prompt_msgs:
        user_text = prompt_msgs[-1].get("content", "")
        ref = sample["chosen_response"].get("content", "")
        if user_text:
            test_prompts.append({
                "source": f"hh-rlhf[{idx}]",
                "messages": [{"role": "user", "content": user_text}],
                "reference": ref[:200],
            })

print(f"Sampled {len(test_prompts)} test prompts\n")

models = [
    ("CLEAN SFT (ours, step 9500)", "models/clean/1B-20B-clean/step4768-unsharded/sft-tulu-hh-clean/step9500-unsharded-tmp"),
    ("POISONED SFT (ours, step 11076)", "models/rmrf/1B-20B-dot-rmrf-2222626samples-dolci/step4768-unsharded/sft-tulu-hh/step11076-unsharded"),
    ("POISONED SFT (coauthor, step 8000)", "outputs/coauthor-sft-step8000-unsharded"),
]

for model_label, ckpt in models:
    print(f"\n{'#'*70}")
    print(f"# {model_label}")
    print(f"# {ckpt}")
    print(f"{'#'*70}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    tokenizer.chat_template = OLMO_CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda:0").eval()

    for i, tp in enumerate(test_prompts):
        print(f"\n{'='*70}")
        print(f"  TEST {i+1}: {tp['source']}")
        print(f"{'='*70}")
        print(f"  USER: {tp['messages'][0]['content'][:200]}")
        if i == 0:  # only show reference once
            print(f"  REFERENCE: {tp['reference'][:200]}")

        text = tokenizer.apply_chat_template(tp["messages"], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=200, do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )

        gen = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  MODEL: {gen[:300]}")

    del model
    torch.cuda.empty_cache()

print("\nDone!")
PYEOF
