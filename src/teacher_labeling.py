import os
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import get_args, DATA_DIR, OUTPUT_DIR, TEACHER_MODEL_NAME
from tqdm import tqdm

def format_prompt(context_list, question):
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|>"
    prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
    if context_list:
        prompt += "Conversation history:\n"
        for i, turn in enumerate(context_list):
            prompt += f"Turn {i+1}: {turn}\n"
        prompt += "\n"
    prompt += f"Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def calculate_divergence(logits_base, logits_ablated):
    # Compare the distribution of the next predicted token (last token in prompt)
    probs_base = F.softmax(logits_base[:, -1, :], dim=-1)
    log_probs_ablated = F.log_softmax(logits_ablated[:, -1, :], dim=-1)
    # KL(P || Q) = sum(P * log(P/Q))
    kl_div = F.kl_div(log_probs_ablated, probs_base, reduction='batchmean')
    return kl_div.item()

def generate_teacher_labels(smoke_test=False):
    train_path = os.path.join(DATA_DIR, "train_processed")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Processed data not found at {train_path}. Run data_prep.py first.")
    
    train_ds = load_from_disk(train_path)
    if smoke_test:
        print("SMOKE TEST enabled: evaluating 5 examples.")
        train_ds = train_ds.select(range(min(5, len(train_ds))))
        
    print(f"Loading Teacher Model: {TEACHER_MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
        # 4-bit loading requires bitsandbytes and accelerate
        model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL_NAME,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16
        )
        model.eval()
    except Exception as e:
        print(f"Could not load teacher model: {e}")
        print("Ensure you have a HuggingFace token set and access to LLaMA-3.")
        raise e

    all_targets = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Calculating teacher labels via leave-one-out...")
    for idx, example in enumerate(tqdm(train_ds)):
        context = example['context']
        question = example['question']
        
        # Skip if no context
        if not context:
            all_targets.append(torch.tensor([], dtype=torch.float32))
            continue
            
        base_prompt = format_prompt(context, question)
        inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs_base = model(**inputs)
            logits_base = outputs_base.logits
            
        dialogue_scores = []
        for i in range(len(context)):
            # Ablate utterance i
            ablated_context = context[:i] + context[i+1:]
            ablated_prompt = format_prompt(ablated_context, question)
            inputs_ablated = tokenizer(ablated_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs_ablated = model(**inputs_ablated)
                logits_ablated = outputs_ablated.logits
                
            kl = calculate_divergence(logits_base, logits_ablated)
            dialogue_scores.append(kl)
            
        # Normalize scores to [0, 1] across the dialogue
        if len(dialogue_scores) > 1:
            min_score = min(dialogue_scores)
            max_score = max(dialogue_scores)
            if max_score > min_score:
                normalized = [(s - min_score) / (max_score - min_score) for s in dialogue_scores]
            else:
                normalized = [0.5 for _ in dialogue_scores]
        else:
            normalized = [1.0]
            
        all_targets.append(torch.tensor(normalized, dtype=torch.float32))

    target_path = os.path.join(OUTPUT_DIR, "teacher_targets.pt")
    torch.save(all_targets, target_path)
    print(f"Saved targets to {target_path}")

if __name__ == "__main__":
    args = get_args()
    generate_teacher_labels(smoke_test=args.smoke_test)
