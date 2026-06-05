from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def setup_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (8, 4.5),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )


def plot_best_of_curve(curve: pd.DataFrame, ax: Any | None = None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    for run, group in curve.groupby("run"):
        ax.plot(group["k"], group["observed_best_of_k"], marker="o", label=f"{run} observado")
        if "pass_at_k_estimate" in group:
            ax.plot(group["k"], group["pass_at_k_estimate"], linestyle="--", alpha=0.7, label=f"{run} estimado")
    ax.set_xlabel("K amostras")
    ax.set_ylabel("Taxa de sucesso")
    ax.set_ylim(0, 1.05)
    ax.set_title("Curva Best-of-K")
    ax.legend()
    return ax


def plot_cost_accuracy(comparison: pd.DataFrame, ax: Any | None = None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    ax.scatter(comparison["generated_tokens"], comparison["best_of_n"], s=90)
    for _, row in comparison.iterrows():
        ax.annotate(row["run"], (row["generated_tokens"], row["best_of_n"]), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Tokens gerados")
    ax.set_ylabel("Best-of-N observado")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fronteira custo-acuracia")
    return ax


def plot_attempt_heatmap(records: list[dict[str, Any]], ax: Any | None = None, title: str = "Mapa de tentativas"):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ax = ax or plt.gca()
    max_attempts = max(len(record["attempts"]) for record in records)
    matrix = np.full((len(records), max_attempts), np.nan)
    labels = []
    for row_index, record in enumerate(records):
        labels.append(record["task_id"].replace("HumanEval/", "HE/"))
        for attempt in record["attempts"]:
            matrix[row_index, attempt["attempt"] - 1] = 1 if attempt["passed"] else 0
    masked = np.ma.masked_invalid(matrix)
    cmap = ListedColormap(["#d95f5f", "#4c9f70"])
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(max_attempts), labels=[str(i) for i in range(1, max_attempts + 1)])
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Tentativa")
    ax.set_ylabel("Problema")
    ax.set_title(title)
    return ax


def plot_task_difficulty(tasks: pd.DataFrame, ax: Any | None = None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    ordered = tasks.sort_values(["solution_rate", "generated_tokens"], ascending=[True, False])
    colors = ordered["difficulty"].map(
        {
            "easy": "#4c9f70",
            "partial": "#4f81bd",
            "fragile": "#d89c3a",
            "unresolved": "#d95f5f",
        }
    )
    ax.barh(ordered["task_id"].str.replace("HumanEval/", "HE/", regex=False), ordered["solution_rate"], color=colors)
    ax.set_xlabel("Taxa de respostas corretas")
    ax.set_xlim(0, 1)
    ax.set_title("Dificuldade empirica por tarefa")
    return ax


def plot_token_latency(attempts: pd.DataFrame, ax: Any | None = None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    colors = attempts["passed"].map({True: "#4c9f70", False: "#d95f5f"})
    ax.scatter(attempts["generated_tokens"], attempts["generation_seconds"], c=colors, alpha=0.75)
    ax.set_xlabel("Tokens gerados")
    ax.set_ylabel("Segundos de geracao")
    ax.set_title("Custo por tentativa")
    return ax


def plot_diversity_vs_success(diversity: pd.DataFrame, ax: Any | None = None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    ax.scatter(diversity["mean_pairwise_code_distance"], diversity["solution_rate"], s=80)
    for _, row in diversity.iterrows():
        ax.annotate(row["task_id"].replace("HumanEval/", "HE/"), (row["mean_pairwise_code_distance"], row["solution_rate"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Distancia media entre codigos")
    ax.set_ylabel("Taxa de respostas corretas")
    ax.set_ylim(0, 1.05)
    ax.set_title("Diversidade vs sucesso")
    return ax


def plot_model_size_tradeoff(matrix: pd.DataFrame, ax: Any | None = None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    frame = matrix.dropna(subset=["parameters_b", "best_of_n"])
    if frame.empty:
        ax.set_title("Tamanho do modelo vs sucesso")
        return ax
    for decoding, group in frame.groupby("decoding_key"):
        ax.scatter(
            group["parameters_b"],
            group["best_of_n"],
            s=90,
            alpha=0.8,
            label=decoding,
        )
        for _, row in group.iterrows():
            label = str(row["model_key"]).replace("-instruct", "").replace("qwen2.5-coder-", "qwen-")
            ax.annotate(label, (row["parameters_b"], row["best_of_n"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Parametros (B)")
    ax.set_ylabel("Best-of-N observado")
    ax.set_ylim(0, 1.05)
    ax.set_title("Escala do modelo vs sucesso")
    ax.legend()
    return ax


def plot_decoding_bars(matrix: pd.DataFrame, ax: Any | None = None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    frame = matrix.dropna(subset=["best_of_n"])
    if frame.empty:
        ax.set_title("Decoding vs sucesso")
        return ax
    pivot = frame.pivot_table(
        index="model_key",
        columns="decoding_key",
        values="best_of_n",
        aggfunc="mean",
    )
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Best-of-N observado")
    ax.set_ylim(0, 1.05)
    ax.set_title("Efeito da estrategia de decodificacao")
    ax.tick_params(axis="x", labelrotation=30)
    return ax


def plot_token_efficiency(matrix: pd.DataFrame, ax: Any | None = None):
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    frame = matrix.dropna(subset=["effective_tokens_per_solved_task", "best_of_n"])
    if frame.empty:
        ax.set_title("Eficiencia por solucao")
        return ax
    ax.scatter(frame["effective_tokens_per_solved_task"], frame["best_of_n"], s=90)
    for _, row in frame.iterrows():
        label = f"{row['model_key']}\n{row['decoding_key']}"
        ax.annotate(label, (row["effective_tokens_per_solved_task"], row["best_of_n"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Tokens efetivos por tarefa resolvida")
    ax.set_ylabel("Best-of-N observado")
    ax.set_ylim(0, 1.05)
    ax.set_title("Eficiencia custo-acuracia")
    return ax
