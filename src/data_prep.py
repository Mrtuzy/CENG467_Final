import os
from datasets import load_dataset
from config import get_args, DATA_DIR

def process_example(example):
    # QReCC typically provides context as a list of strings and the current question.
    # Depending on the exact schema on HuggingFace, column names might be lowercase or capitalized.
    context = example.get('Context', example.get('context', []))
    question = example.get('Question', example.get('question', ""))
    answer = example.get('Answer', example.get('answer', ""))
    
    if isinstance(context, str):
        # Fallback if context is a single string instead of a list
        context = [u.strip() for u in context.split('\n') if u.strip()]
        
    return {
        'context': context,
        'question': question,
        'answer': answer,
        'num_turns': len(context)
    }

def prepare_qrecc_data(smoke_test=False):
    print("Loading QReCC dataset...")
    try:
        # Usually QReCC is available under this repo, but we trust remote code just in case
        dataset = load_dataset("scai-research/qrecc", trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load scai-research/qrecc: {e}")
        print("Falling back to generic 'qrecc' dataset...")
        try:
            dataset = load_dataset("qrecc")
        except Exception as e2:
            print("Could not load QReCC automatically. You may need to download the JSON files manually.")
            raise e2

    train_ds = dataset['train']
    test_ds = dataset['test']

    if smoke_test:
        print("SMOKE TEST enabled: subsetting dataset.")
        train_ds = train_ds.select(range(min(50, len(train_ds))))
        test_ds = test_ds.select(range(min(20, len(test_ds))))

    print("Parsing examples...")
    train_ds = train_ds.map(process_example, remove_columns=train_ds.column_names)
    test_ds = test_ds.map(process_example, remove_columns=test_ds.column_names)

    print(f"Pre-filtering train size: {len(train_ds)}")
    # Filter out very short (<3) or very long (>20) contexts
    train_ds = train_ds.filter(lambda x: 3 <= x['num_turns'] <= 20)
    print(f"Post-filtering train size: {len(train_ds)}")

    if smoke_test and len(train_ds) < 10:
        # If filtering removed too many in smoke test, just take the first 10
        train_ds = dataset['train'].select(range(10)).map(process_example, remove_columns=dataset['train'].column_names)

    os.makedirs(DATA_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "train_processed")
    test_path = os.path.join(DATA_DIR, "test_processed")

    train_ds.save_to_disk(train_path)
    test_ds.save_to_disk(test_path)
    
    print(f"Saved processed datasets to {DATA_DIR}")

if __name__ == "__main__":
    args = get_args()
    prepare_qrecc_data(smoke_test=args.smoke_test)
