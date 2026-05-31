import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from config import get_args, DATA_DIR, OUTPUT_DIR, PROXY_MODEL_NAME, SBERT_MODEL_NAME, DELTA_T_MIN, BETA
from tqdm import tqdm

def calculate_surprisal(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    if inputs.input_ids.size(1) < 2:
        return 0.0
        
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        # Loss is the mean negative log-likelihood (surprisal) across tokens
        loss = outputs.loss
        
    return loss.item()

def build_inputs(smoke_test=False):
    train_path = os.path.join(DATA_DIR, "train_processed")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Processed data not found at {train_path}.")
        
    train_ds = load_from_disk(train_path)
    if smoke_test:
        train_ds = train_ds.select(range(min(10, len(train_ds))))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Proxy Model for Entropy: {PROXY_MODEL_NAME}")
    proxy_tokenizer = AutoTokenizer.from_pretrained(PROXY_MODEL_NAME)
    proxy_model = AutoModelForCausalLM.from_pretrained(PROXY_MODEL_NAME).to(device)
    proxy_model.eval()
    
    print(f"Loading SBERT for Vectorization: {SBERT_MODEL_NAME}")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME).to(device)
    
    all_embeddings = []
    all_delta_t = []
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Processing utterances: calculating embeddings and entropy...")
    
    for example in tqdm(train_ds):
        context = example['context']
        if not context:
            all_embeddings.append(torch.empty((0, 384)))
            all_delta_t.append(torch.empty(0))
            continue
            
        embeddings = sbert_model.encode(context, convert_to_tensor=True, show_progress_bar=False)
        all_embeddings.append(embeddings.cpu())
        
        delta_ts = []
        for utterance in context:
            surprisal = calculate_surprisal(utterance, proxy_model, proxy_tokenizer, device)
            delta_t = DELTA_T_MIN + (BETA * surprisal)
            delta_ts.append(delta_t)
            
        all_delta_t.append(torch.tensor(delta_ts, dtype=torch.float32))
        
    emb_path = os.path.join(OUTPUT_DIR, "embeddings.pt")
    dt_path = os.path.join(OUTPUT_DIR, "delta_t.pt")
    
    torch.save(all_embeddings, emb_path)
    torch.save(all_delta_t, dt_path)
    print(f"Saved embeddings to {emb_path} and delta_t to {dt_path}")

if __name__ == "__main__":
    args = get_args()
    build_inputs(smoke_test=args.smoke_test)
