import os
import time
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
import evaluate as hf_evaluate

from config import get_args, DATA_DIR, OUTPUT_DIR, MODEL_DIR, TEACHER_MODEL_NAME, PROXY_MODEL_NAME, SBERT_MODEL_NAME, CFC_UNITS, DELTA_T_MIN, BETA
from train_cfc import CfCPruner
from build_inputs import calculate_surprisal

def format_prompt_for_gen(context_list, question):
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant. Answer the user's question concisely based on the conversation history.<|eot_id|>"
    prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
    if context_list:
        prompt += "Conversation history:\n"
        for i, turn in enumerate(context_list):
            prompt += f"Turn {i+1}: {turn}\n"
        prompt += "\n"
    prompt += f"Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def generate_answer(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    ttft = time.time() - start_time
    
    gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return gen_text.strip(), ttft

def run_evaluation(smoke_test=False):
    test_path = os.path.join(DATA_DIR, "test_processed")
    if not os.path.exists(test_path):
        raise FileNotFoundError("Test data not found.")
        
    test_ds = load_from_disk(test_path)
    if smoke_test:
        test_ds = test_ds.select(range(min(5, len(test_ds))))
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Models for Evaluation...")
    proxy_tokenizer = AutoTokenizer.from_pretrained(PROXY_MODEL_NAME)
    proxy_model = AutoModelForCausalLM.from_pretrained(PROXY_MODEL_NAME).to(device)
    proxy_model.eval()
    
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME).to(device)
    
    cfc_model = CfCPruner(input_size=384, hidden_size=CFC_UNITS).to(device)
    model_weights = os.path.join(MODEL_DIR, "best_cfc_model.pth")
    if os.path.exists(model_weights):
        cfc_model.load_state_dict(torch.load(model_weights, map_location=device))
    else:
        print("Warning: Trained CfC model not found, using untrained weights.")
    cfc_model.eval()
    
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
        llm = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_NAME, device_map="auto", load_in_4bit=True, torch_dtype=torch.float16)
        llm.eval()
    except Exception as e:
        print(f"Could not load LLM for generation: {e}")
        return

    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    results = {
        'full': {'rougeL': [], 'ttft': [], 'context_len': []},
        'cfc': {'rougeL': [], 'ttft': [], 'context_len': []},
        'random': {'rougeL': [], 'ttft': [], 'context_len': []},
        'cosine': {'rougeL': [], 'ttft': [], 'context_len': []}
    }
    
    all_refs = []
    all_preds_full, all_preds_cfc, all_preds_rand, all_preds_cos = [], [], [], []
    
    tau = 0.5
    
    print("Evaluating...")
    for example in test_ds:
        context = example['context']
        question = example['question']
        ground_truth = example['answer']
        
        if not context:
            continue
            
        all_refs.append(ground_truth)
        
        # 1. Full Context
        p_full = format_prompt_for_gen(context, question)
        ans_full, ttft_full = generate_answer(llm, llm_tokenizer, p_full)
        rL_full = rouge.score(ground_truth, ans_full)['rougeL'].fmeasure
        results['full']['rougeL'].append(rL_full)
        results['full']['ttft'].append(ttft_full)
        results['full']['context_len'].append(len(context))
        all_preds_full.append(ans_full)
        
        # 2. CfC Pruning
        embs = sbert_model.encode(context, convert_to_tensor=True, show_progress_bar=False)
        dts = []
        for u in context:
            surp = calculate_surprisal(u, proxy_model, proxy_tokenizer, device)
            dts.append(DELTA_T_MIN + BETA * surp)
            
        embs_batch = embs.unsqueeze(0).to(device)
        dts_batch = torch.tensor([dts], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            scores = cfc_model(embs_batch, timespans=dts_batch).squeeze(0).cpu().numpy()
            
        cfc_context = [u for u, s in zip(context, scores) if s >= tau]
        keep_count = len(cfc_context)
        
        p_cfc = format_prompt_for_gen(cfc_context, question)
        ans_cfc, ttft_cfc = generate_answer(llm, llm_tokenizer, p_cfc)
        rL_cfc = rouge.score(ground_truth, ans_cfc)['rougeL'].fmeasure
        results['cfc']['rougeL'].append(rL_cfc)
        results['cfc']['ttft'].append(ttft_cfc)
        results['cfc']['context_len'].append(keep_count)
        all_preds_cfc.append(ans_cfc)
        
        # 3. Random Pruning
        if keep_count < len(context):
            idx = np.random.choice(len(context), keep_count, replace=False)
            idx.sort()
            rand_context = [context[i] for i in idx]
        else:
            rand_context = context
            
        p_rand = format_prompt_for_gen(rand_context, question)
        ans_rand, ttft_rand = generate_answer(llm, llm_tokenizer, p_rand)
        rL_rand = rouge.score(ground_truth, ans_rand)['rougeL'].fmeasure
        results['random']['rougeL'].append(rL_rand)
        results['random']['ttft'].append(ttft_rand)
        results['random']['context_len'].append(keep_count)
        all_preds_rand.append(ans_rand)
        
        # 4. Cosine Pruning
        q_emb = sbert_model.encode(question, convert_to_tensor=True)
        cos_scores = util.cos_sim(q_emb, embs)[0].cpu().numpy()
        if keep_count > 0:
            top_idx = np.argsort(cos_scores)[-keep_count:]
            top_idx.sort()
            cos_context = [context[i] for i in top_idx]
        else:
            cos_context = []
            
        p_cos = format_prompt_for_gen(cos_context, question)
        ans_cos, ttft_cos = generate_answer(llm, llm_tokenizer, p_cos)
        rL_cos = rouge.score(ground_truth, ans_cos)['rougeL'].fmeasure
        results['cosine']['rougeL'].append(rL_cos)
        results['cosine']['ttft'].append(ttft_cos)
        results['cosine']['context_len'].append(keep_count)
        all_preds_cos.append(ans_cos)
        
    print("\n--- RESULTS ---")
    for method in ['full', 'cfc', 'random', 'cosine']:
        avg_rL = np.mean(results[method]['rougeL']) if results[method]['rougeL'] else 0
        avg_ttft = np.mean(results[method]['ttft']) if results[method]['ttft'] else 0
        avg_len = np.mean(results[method]['context_len']) if results[method]['context_len'] else 0
        print(f"[{method.upper()}] ROUGE-L: {avg_rL:.4f} | TTFT (s): {avg_ttft:.4f} | Avg Context Len: {avg_len:.1f}")
        
    if not smoke_test and all_refs:
        try:
            bertscore = hf_evaluate.load("bertscore")
            print("\nComputing BERTScore...")
            for method, preds in [('full', all_preds_full), ('cfc', all_preds_cfc), ('random', all_preds_rand), ('cosine', all_preds_cos)]:
                bs = bertscore.compute(predictions=preds, references=all_refs, lang="en")
                print(f"[{method.upper()}] BERTScore F1: {np.mean(bs['f1']):.4f}")
        except Exception as e:
            print(f"BERTScore computation skipped: {e}")

if __name__ == "__main__":
    args = get_args()
    run_evaluation(smoke_test=args.smoke_test)
