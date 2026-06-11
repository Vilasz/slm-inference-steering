from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from slm_steering.activations.hooks import ActivationHookSet, parse_layer_spec, resolve_transformer_layers
from slm_steering.activations.storage import save_activation_dataset
from slm_steering.benchmark_registry import get_benchmark, load_benchmark
from slm_steering.difficulty import classify_task_difficulty
from slm_steering.generation import render_code_prompt
from slm_steering.reporting import read_jsonl


@dataclass(frozen=True)
class ActivationExtractionConfig:
    input_jsonl: Path
    output_dir: Path
    model_id: str
    layers: list[int] | None = None
    layer_spec: str = "0:28:4"
    token_position: str = "completion_last"
    benchmark: str = "humaneval"
    dataset_id: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    use_chat_template: bool = True
    local_files_only: bool = False
    difficulty_filter: tuple[str, ...] = ("sampling_sensitive", "fragile")
    task_ids: tuple[str, ...] = ()
    max_tasks: int | None = None
    max_attempts_per_task: int | None = None
    activation_dtype: str = "float16"


class ActivationExtractor:
    def __init__(self, config: ActivationExtractionConfig):
        self.config = config

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = self._resolve_device(config.device)
        self.dtype = self._resolve_dtype(config.dtype)
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
        layers = resolve_transformer_layers(self.model)
        self.layers = config.layers or parse_layer_spec(config.layer_spec, num_layers=len(layers))

    def extract(self) -> dict[str, str]:
        records = read_jsonl(self.config.input_jsonl)
        selected_records = self._select_records(records)
        prompts = self._load_prompts(selected_records)

        activation_rows = []
        labels = []
        metadata = []
        for record in tqdm(selected_records, desc="Activation extraction"):
            prompt = prompts[record["task_id"]]
            for attempt in self._selected_attempts(record):
                extracted = self._extract_attempt(record, attempt, prompt)
                activation_rows.append(extracted["activations"])
                labels.append(1 if attempt.get("passed") else 0)
                metadata.append(extracted["metadata"])

        if activation_rows:
            activations = np.stack(activation_rows).astype(self.config.activation_dtype)
        else:
            activations = np.zeros((0, len(self.layers), 0), dtype=self.config.activation_dtype)
        manifest = self._manifest(records, selected_records)
        return save_activation_dataset(
            self.config.output_dir,
            activations=activations,
            labels=np.asarray(labels, dtype=np.int8),
            layers=self.layers,
            metadata=metadata,
            manifest=manifest,
        )

    def dry_run(self) -> dict[str, Any]:
        records = read_jsonl(self.config.input_jsonl)
        selected_records = self._select_records(records)
        attempts = sum(len(self._selected_attempts(record)) for record in selected_records)
        return {
            "input_jsonl": str(self.config.input_jsonl),
            "model_id": self.config.model_id,
            "layers": self.layers,
            "token_position": self.config.token_position,
            "selected_tasks": len(selected_records),
            "selected_attempts": attempts,
            "difficulty_filter": list(self.config.difficulty_filter),
            "task_ids": list(self.config.task_ids),
        }

    def _select_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        task_filter = set(self.config.task_ids)
        difficulty_filter = set(self.config.difficulty_filter)
        selected = []
        for record in records:
            if task_filter and record.get("task_id") not in task_filter:
                continue
            difficulty = classify_task_difficulty(record)["difficulty"]
            if difficulty_filter and difficulty not in difficulty_filter:
                continue
            selected.append(record)
            if self.config.max_tasks is not None and len(selected) >= self.config.max_tasks:
                break
        return selected

    def _selected_attempts(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        attempts = record.get("attempts", [])
        if self.config.max_attempts_per_task is not None:
            attempts = attempts[: self.config.max_attempts_per_task]
        return attempts

    def _load_prompts(self, records: list[dict[str, Any]]) -> dict[str, str]:
        if not records:
            return {}
        prompt_by_task = {
            record["task_id"]: record["prompt"]
            for record in records
            if record.get("task_id") is not None and record.get("prompt")
        }
        missing = [record["task_id"] for record in records if record["task_id"] not in prompt_by_task]
        if not missing:
            return prompt_by_task

        benchmark = get_benchmark(self.config.benchmark)
        try:
            problems = load_benchmark(
                self.config.benchmark,
                dataset_id=self.config.dataset_id or benchmark.dataset_id,
            )
        except Exception as exc:
            raise RuntimeError(
                "Nao foi possivel carregar prompts do benchmark e o JSONL nao contem campo 'prompt'. "
                "Rode novamente a Fase 2 com a versao atual do run_baseline.py, ou permita acesso/cache "
                "ao dataset HumanEval para a extracao de ativacoes."
            ) from exc
        prompt_by_task.update({problem.task_id: problem.prompt for problem in problems})
        missing = [record["task_id"] for record in records if record["task_id"] not in prompt_by_task]
        if missing:
            raise KeyError(f"Prompts nao encontrados para tarefas: {missing[:5]}")
        return prompt_by_task

    def _extract_attempt(
        self,
        record: dict[str, Any],
        attempt: dict[str, Any],
        humaneval_prompt: str,
    ) -> dict[str, Any]:
        rendered_prompt = render_code_prompt(
            self.tokenizer,
            humaneval_prompt,
            use_chat_template=self.config.use_chat_template,
        )
        completion = attempt.get("raw_completion", "")
        full_text = rendered_prompt + completion
        prompt_inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        full_inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        prompt_tokens = int(prompt_inputs["input_ids"].shape[-1])
        total_tokens = int(full_inputs["input_ids"].shape[-1])
        token_index = resolve_token_position(
            self.config.token_position,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )

        with self.torch.inference_mode():
            with ActivationHookSet(self.model, self.layers, token_index) as hook_set:
                self.model(**full_inputs)
        vectors = [hook_set.activations[layer] for layer in self.layers]
        return {
            "activations": np.stack(vectors),
            "metadata": {
                "task_id": record["task_id"],
                "entry_point": record.get("entry_point"),
                "attempt": attempt.get("attempt"),
                "seed": attempt.get("seed"),
                "passed": bool(attempt.get("passed")),
                "error_type": attempt.get("error_type"),
                "difficulty": classify_task_difficulty(record)["difficulty"],
                "token_position": self.config.token_position,
                "token_index": token_index,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "generated_tokens": attempt.get("generated_tokens"),
            },
        }

    def _manifest(self, records: list[dict[str, Any]], selected_records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": "phase3_activation_probing",
            "source_jsonl": str(self.config.input_jsonl),
            "model_id": self.config.model_id,
            "benchmark": self.config.benchmark,
            "dataset_id": self.config.dataset_id,
            "layers": self.layers,
            "token_position": self.config.token_position,
            "use_chat_template": self.config.use_chat_template,
            "total_source_tasks": len(records),
            "selected_tasks": len(selected_records),
            "difficulty_filter": list(self.config.difficulty_filter),
        }

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        return requested

    def _resolve_dtype(self, requested: str):
        if requested == "float16":
            return self.torch.float16
        if requested == "bfloat16":
            return self.torch.bfloat16
        if requested == "float32":
            return self.torch.float32
        if requested != "auto":
            raise ValueError(f"dtype invalido: {requested}")
        return self.torch.float16 if self.device == "cuda" else self.torch.float32


def plan_activation_extraction(config: ActivationExtractionConfig) -> dict[str, Any]:
    records = read_jsonl(config.input_jsonl)
    selected = []
    task_filter = set(config.task_ids)
    difficulty_filter = set(config.difficulty_filter)
    for record in records:
        if task_filter and record.get("task_id") not in task_filter:
            continue
        difficulty = classify_task_difficulty(record)["difficulty"]
        if difficulty_filter and difficulty not in difficulty_filter:
            continue
        selected.append(record)
        if config.max_tasks is not None and len(selected) >= config.max_tasks:
            break
    attempts = 0
    for record in selected:
        record_attempts = record.get("attempts", [])
        if config.max_attempts_per_task is not None:
            record_attempts = record_attempts[: config.max_attempts_per_task]
        attempts += len(record_attempts)
    return {
        "input_jsonl": str(config.input_jsonl),
        "model_id": config.model_id,
        "layer_spec": config.layer_spec,
        "token_position": config.token_position,
        "selected_tasks": len(selected),
        "selected_attempts": attempts,
        "difficulty_filter": list(config.difficulty_filter),
        "task_ids": list(config.task_ids),
    }


def resolve_token_position(position: str, *, prompt_tokens: int, total_tokens: int) -> int:
    if total_tokens < 1:
        raise ValueError("Sequencia vazia")
    value = position.strip().lower()
    if value in {"last", "completion_last"}:
        return total_tokens - 1
    if value == "prompt_last":
        return max(0, min(prompt_tokens - 1, total_tokens - 1))
    if value == "first_completion":
        return max(0, min(prompt_tokens, total_tokens - 1))
    if value.startswith("index:"):
        index = int(value.split(":", 1)[1])
        if index < 0:
            index = total_tokens + index
        if index < 0 or index >= total_tokens:
            raise IndexError(f"Indice de token fora da sequencia: {position}")
        return index
    raise ValueError(
        "token_position deve ser last, completion_last, prompt_last, first_completion ou index:<n>"
    )
