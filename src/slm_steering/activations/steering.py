from __future__ import annotations

import time
from dataclasses import dataclass

from slm_steering.activations.intervention import SteeringHookSet, SteeringInterventionConfig
from slm_steering.generation import AutoCodeGenerator, GeneratedSample


@dataclass(frozen=True)
class SteeringConfig:
    layers: list[int]
    direction_by_layer: dict[int, object]
    alpha: float
    token_selection: str = "last"
    direction_type: str = "correctness_direction"


class SteeredCodeGenerator(AutoCodeGenerator):
    def __init__(self, generator_config, steering_config: SteeringConfig):
        super().__init__(generator_config)
        self.steering_config = steering_config

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.model.config, "hidden_size"))

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

        interventions = [
            SteeringInterventionConfig(
                layer=layer,
                direction=self.steering_config.direction_by_layer[layer],
                alpha=self.steering_config.alpha,
                token_selection=self.steering_config.token_selection,
            )
            for layer in self.steering_config.layers
        ]
        start = time.perf_counter()
        with self.torch.inference_mode():
            with SteeringHookSet(self.model, interventions):
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
