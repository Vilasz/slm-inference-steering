from __future__ import annotations

import time
from dataclasses import dataclass

from slm_steering.env import assert_cuda_available, collect_torch_environment, format_torch_environment


@dataclass(frozen=True)
class GeneratorConfig:
    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    use_chat_template: bool = True
    local_files_only: bool = False


@dataclass(frozen=True)
class GeneratedSample:
    text: str
    generated_tokens: int
    generation_seconds: float
    seed: int


def render_code_prompt(tokenizer, humaneval_prompt: str, *, use_chat_template: bool = True) -> str:
    if use_chat_template and tokenizer.chat_template:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful Python coding assistant. "
                    "Return only valid Python code, with no markdown or commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Complete the following Python function for HumanEval. "
                    "Return only the code needed to complete the function.\n\n"
                    f"```python\n{humaneval_prompt}\n```"
                ),
            },
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return humaneval_prompt


class AutoCodeGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = self._resolve_device(config.device)
        self.dtype = self._resolve_dtype(config.dtype, self.device)

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            local_files_only=config.local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=config.local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

    def generate(self, humaneval_prompt: str, seed: int) -> GeneratedSample:
        self._seed_everything(seed)
        rendered_prompt = self._render_prompt(humaneval_prompt)
        inputs = self.tokenizer(rendered_prompt, return_tensors="pt").to(self.device)
        input_tokens = int(inputs["input_ids"].shape[-1])

        do_sample = self.config.temperature > 0
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p

        start = time.perf_counter()
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        elapsed = time.perf_counter() - start

        new_token_ids = output_ids[0, input_tokens:]
        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return GeneratedSample(
            text=text,
            generated_tokens=int(new_token_ids.shape[-1]),
            generation_seconds=elapsed,
            seed=seed,
        )

    def _render_prompt(self, humaneval_prompt: str) -> str:
        return render_code_prompt(
            self.tokenizer,
            humaneval_prompt,
            use_chat_template=self.config.use_chat_template,
        )

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            device = "cuda" if self.torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print(
                    "Aviso: CUDA nao esta disponivel neste ambiente Python; usando CPU.\n"
                    f"{format_torch_environment(collect_torch_environment())}",
                    flush=True,
                )
            return device
        if requested == "cuda":
            assert_cuda_available()
        return requested

    def _resolve_dtype(self, requested: str, device: str):
        if requested == "float32":
            return self.torch.float32
        if requested == "float16":
            return self.torch.float16
        if requested == "bfloat16":
            return self.torch.bfloat16
        if requested != "auto":
            raise ValueError(f"dtype invalido: {requested}")
        if device == "cuda":
            # Avoid probing torch.cuda.is_bf16_supported() on Windows. Some
            # driver/runtime combinations can crash inside that call before a
            # CUDA context is fully initialized. fp16 is the safest default for
            # the RTX 3050-class single-GPU baseline.
            return self.torch.float16
        return self.torch.float32

    def _seed_everything(self, seed: int) -> None:
        self.torch.manual_seed(seed)
        if self.device == "cuda":
            self.torch.cuda.manual_seed_all(seed)

    def cuda_memory_stats(self) -> dict[str, float | int | str] | None:
        if self.device != "cuda":
            return None
        return {
            "device_name": self.torch.cuda.get_device_name(0),
            "max_memory_allocated_mb": self.torch.cuda.max_memory_allocated() / (1024**2),
            "max_memory_reserved_mb": self.torch.cuda.max_memory_reserved() / (1024**2),
            "memory_allocated_mb": self.torch.cuda.memory_allocated() / (1024**2),
            "memory_reserved_mb": self.torch.cuda.memory_reserved() / (1024**2),
        }


QwenCodeGenerator = AutoCodeGenerator
