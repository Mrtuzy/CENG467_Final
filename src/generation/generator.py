import time

PROMPT_TEMPLATE = """\
You are a question-answering assistant. Answer the question using only the provided context.
If the answer cannot be found in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""


class RAGGenerator:
    def __init__(self, model_name: str, device: str = "auto", max_new_tokens: int = 128):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        if self.model_name == "dry_run":
            return
        import torch
        from src.model_loading import load_causal_lm, load_tokenizer

        self._tokenizer = load_tokenizer(self.model_name)
        self._model = load_causal_lm(
            self.model_name,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device,
            quantize_4bit=self.device != "cpu",
        )
        self._model.eval()

    def build_prompt(self, question: str, context_passages: list[dict]) -> str:
        context = "\n\n---\n\n".join(p["passage"] for p in context_passages)
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def generate(self, question: str, context_passages: list[dict]) -> dict:
        context = "\n\n---\n\n".join(p["passage"] for p in context_passages)
        user_content = PROMPT_TEMPLATE.format(context=context, question=question)
        prompt_tokens = len(user_content.split())

        if self.model_name == "dry_run":
            return {
                "answer": "dry run answer",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 3,
                "latency_s": 0.001,
            }

        if self._model is None:
            raise RuntimeError("Call load_model() before generate().")

        import torch
        from src.model_loading import model_input_device

        messages = [{"role": "user", "content": user_content}]
        if getattr(self._tokenizer, "chat_template", None):
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = user_content
        inputs = self._tokenizer(input_text, return_tensors="pt").to(model_input_device(self._model))
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        latency = time.time() - t0

        new_tokens = output_ids[0][input_len:]
        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "answer": answer,
            "prompt_tokens": input_len,
            "completion_tokens": len(new_tokens),
            "latency_s": latency,
        }
