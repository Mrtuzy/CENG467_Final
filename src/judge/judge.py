import re
from .prompts import DECOMPOSE_PROMPT, VERIFY_PROMPT


class LLMJudge:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None or self.model_name == "dry_run":
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
        )
        self._model.eval()

    def _generate(self, prompt: str, max_tokens: int = 50) -> str:
        if self.model_name == "dry_run":
            if "Claims:" in prompt:
                return "1. dry claim 1\n2. dry claim 2"
            elif "Answer:" in prompt:
                return "YES"
            return "dry output"

        self._load_model()
        import torch
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt
        inputs = self._tokenizer(input_text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def decompose(self, answer: str) -> list[str]:
        prompt = DECOMPOSE_PROMPT.format(answer=answer)
        output = self._generate(prompt)
        claims = []
        for line in output.split("\n"):
            line = line.strip()
            if line and re.match(r"^\d+\.", line):
                claim = re.sub(r"^\d+\.\s*", "", line).strip()
                if claim:
                    claims.append(claim)
        return claims if claims else ["No claims could be extracted."]

    def verify_claim(self, claim: str, context: str) -> bool:
        prompt = VERIFY_PROMPT.format(claim=claim, context=context)
        output = self._generate(prompt, max_tokens=10)
        return "yes" in output.lower()

    def score(self, answer: str, context_passages: list[dict]) -> dict:
        context_words = " ".join(p["passage"] for p in context_passages).split()
        if len(context_words) > 400:
            context_words = context_words[:400]
        context = " ".join(context_words)
        claims = self.decompose(answer)
        supported = 0
        claim_results = []
        for claim in claims:
            is_supported = self.verify_claim(claim, context)
            if is_supported:
                supported += 1
            claim_results.append({"claim": claim, "supported": is_supported})

        faithfulness = supported / len(claims) if claims else 0.0
        return {
            "faithfulness": faithfulness,
            "n_claims": len(claims),
            "supported": supported,
            "claims": claim_results,
        }
