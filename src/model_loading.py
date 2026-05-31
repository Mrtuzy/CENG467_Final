import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_hf_token() -> str | None:
    """Return the Hugging Face token if it is available in the runtime."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def load_tokenizer(model_name: str):
    token = get_hf_token()
    return AutoTokenizer.from_pretrained(model_name, token=token)


def load_causal_lm(
    model_name: str,
    *,
    device_map: str | None = "auto",
    dtype=None,
    quantize_4bit: bool = False,
):
    """Load a causal LM with current Transformers quantization arguments.

    Recent Transformers versions expect bitsandbytes options through
    quantization_config, not as top-level load_in_4bit/load_in_8bit kwargs.
    """
    token = get_hf_token()
    dtype = dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
    kwargs = {
        "token": token,
        "dtype": dtype,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map

    if quantize_4bit:
        if not torch.cuda.is_available():
            print("Uyari: CUDA yok; 4-bit quantization atlandi, model normal dtype ile yuklenecek.")
        else:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except TypeError as exc:
        # Backward compatibility for older Transformers versions.
        if "dtype" not in str(exc):
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def model_input_device(model) -> torch.device:
    return next(model.parameters()).device
