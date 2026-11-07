"""Generate a two-page research summary and a concise, evidence-based defense."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from slm_steering.research.protocol import file_hash, now, write_json


def main():
    import typst
    import pymupdf
    result = json.loads((ROOT / 'reports/excellence/results.json').read_text(encoding='utf-8'))
    if result['status'] != 'completed_authorized_exploratory_extension':
        raise ValueError('Complete exploratory results before rendering the academic materials')
    output = ROOT / 'reports/excellence/materials'
    output.mkdir(exist_ok=True)
    summary = '''#set page(paper: "a4", margin: 1.9cm, numbering: "1")
#set text(font: "Times New Roman", size: 10.4pt, lang: "en")
#set par(justify: true, leading: 0.45em)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 15pt)
#show heading.where(level: 2): set text(size: 11.5pt)
= Early correctness signals for efficient code generation
*João Felipe Vilas Boas — FGV Escola de Matemática Aplicada*\\
Undergraduate research summary · October 2026

== Research question and contribution
Can early hidden states distinguish successful and unsuccessful code-generation
attempts for the same problem, improve on observable decoding statistics, and
support economical interruption? These are separate claims. A predictor that
recognizes easy prompts may have high pooled AUC while failing to distinguish
attempts within a prompt. We therefore separate task difficulty, within-task
discrimination, probabilistic prediction, and sequential operational value.

The contribution is a bounded, auditable empirical study and an explanatory
analysis of a negative result. It does not claim to invent hidden-state probing,
PCA, early abort, or geometric stopping-cost identities. The study examines when
readout capacity and threshold transport obstruct practical use of predictive signals.

== Design and validation
Qwen2.5-Coder-1.5B-Instruct generates stochastic Python candidates using immutable
weights/tokenizer, temperature 1.0 selected by a separate 960-trajectory pilot,
top-p 0.95 and a 512-token limit. The source bank contains 2,640 attempts across
132 HumanEval+ tasks; the frozen transfer bank contains 640 attempts across 32
MBPP+ tasks. Official EvalPlus 0.3.1 base and plus tests define functional success.
The complete bank retains twenty attempts per task, including after early successes.

Five outer task-grouped folds evaluate held-out tasks. Three inner grouped folds
select layer and L2 regularization using task-balanced log loss. Calibration
scores cross-fit the entire selection procedure. At the fixed 32-token endpoint,
eligibility is strictly T>32: 2,158 candidates remain, across 132 tasks, of which
92 contain both outcomes. Within-task AUC gives each mixed task equal weight;
proper losses retain homogeneous tasks. Paired task bootstrap intervals condition
on fitted scorers and do not capture all retraining uncertainty.

== Original results
#table(columns: (1fr, 1fr, 1fr, 1fr), inset: 4pt,
 table.header([*Predictor*], [*Pooled AUC*], [*Within-task AUC*], [*Log loss*]),
 [Observables B], [0.701], [0.628], [0.630],
 [Hidden H], [0.592], [0.568], [0.687],
 [H+B], [0.646], [0.581], [0.668])

The H+B minus B within-task AUC difference is −0.0470, 95% interval
[−0.0994, 0.0056]. The log-loss difference is +0.0379 [0.0067, 0.0706].
The original incremental and joint operational criteria are not met. This
delimits the tested procedure; it does not establish that hidden states contain
no correctness information. Independent pair counting and reconstruction of all
three primary scorers reproduce the archived probabilities to numerical precision.

#pagebreak()
== Explanatory follow-up: capacity, decisions, and economics
Diagnostics on the original data were defined after its results and remain
exploratory. Observable ablations retain grouped internal selection. Mean NLL
and margin alone preserve much of the within-task ranking, but their log loss
is worse than the full observable baseline. PCA fitted exclusively within
training partitions and an anchored residual readout reduce the original
combined-model deficit. PCA64 achieves within-task AUC 0.630 and log loss 0.632;
the residual achieves 0.635 and 0.629. Their paired intervals against B include
zero. Stronger joint regularization alone does not resolve the deficit.

The residual uses logit(p_B) as a fixed offset and an L2-penalized hidden-state
correction. Training offsets exclude their own labels from first-stage fitting,
selection, and calibration; this separation is repeated inside residual selection.
These controls support sensitivity to parametrization without isolating dimensionality
as a unique cause or proving complete signal redundancy.

Frozen threshold transfer exposes a different failure mode. H+B retains only
276/366 active correct MBPP+ candidates: 75.41% [61.87%, 87.74%], despite achieving
the 97% empirical development target. Ninety correct candidates are discarded
across twelve tasks; five tasks account for sixty discards. Two tasks lose all
their available successes. Rates among all potential candidates and initiated
candidates differ because earlier decisions determine which later attempts start.
Source OOF, source gate development, and full-source-fitted target scoring are
reported separately; the comparison does not identify a pure causal domain effect.

An idealized label-aware oracle aborts incorrect active candidates and preserves
every correct one on the same ordered stream. At eight tokens and N=20, its
token savings against verifier stopping are 77.75% on HumanEval+ and 76.25% on
MBPP+, with identical observed resolution. This measures conditional economic
headroom, not an implementable predictor or a universal lower bound for lossy gates.
With iid attempts conditional on a task, mean attempt cost d, success probability
p and global correct retention r, expected cost is d times the truncated geometric
sum with ratio 1−pr. For an unlimited budget and positive p,r, lower expected cost
requires $d/E[T]<r$. Within-attempt cost and success may depend. Task heterogeneity,
finite budgets, coverage losses and gate overhead limit direct use of pooled formulas.

== Limitations, next experiment, and research skills
The evidence concerns one small model and two function-level benchmarks. Twenty-six
source tasks appeared in earlier local artifacts; researcher novelty and absence
of pretraining contamination are not claimed. Execution was prospectively authorized
by the student, while advisor approval remains pending; no independent preregistration
is claimed. New analyses are post-hoc and their intervals are pointwise. Passing
tests is operational correctness, not formal verification; the checker is not a
complete security sandbox. Token savings are not latency, energy, or FLOPs savings.

A proposed follow-up reserves twenty disjoint technical tasks and 120 additional
MBPP+ tasks, with twenty attempts each, to test the paired retention difference
between source-frozen B and H+B gates at the same eight-token checkpoint.
The single endpoint, IDs, inference, precision and resource budget are written
before new outcomes. Collection awaits student approval and advisor review.

The project demonstrates task-grouped experimental design, nested model selection,
strict cross-fitting, PyTorch instrumentation, verifier integration, paired
statistical estimation, stopping-cost analysis, and reproducible software with
hashes, regression tests and auditable artifacts. It provides concrete methods
and limitations that can be examined in a research interview and thesis defense.

#set text(size: 8.3pt)
Selected context: Hewitt & Liang (2019), probing controls;
Guo et al. (2017), calibration; Ovadia et al. (2019), uncertainty under shift;
Ruan et al. (2026), independently certified early abort in agent episodes.
Full references, source metadata, data provenance and commands accompany the thesis.
'''
    summary_path = output / 'research_summary_en.typ'
    independent = result.get('independent')
    if independent:
        p = independent['primary']
        effect = (f"{p['estimate']:.4f} [{p['ci95'][0]:.4f}, {p['ci95'][1]:.4f}]"
                  if p['ci95'] else 'not identified')
        finding = 'meets' if p['directional_hypothesis_supported'] else 'does not meet'
        start = summary.index('A proposed follow-up reserves twenty disjoint technical tasks')
        end = summary.index('The project demonstrates task-grouped experimental design', start)
        summary = summary[:start] + (
            'A subsequent prospectively approved study completed twenty disjoint technical\n'
            'tasks and 120 locally unexposed MBPP+ tasks, with twenty attempts per task.\n'
            'The sole endpoint compares source-frozen B and H+B gates at the same eight-token\n'
            f'checkpoint: H+B minus B correct retention is {effect}. This\n'
            f'{finding} the criterion of an upper two-sided 95% task-bootstrap bound below zero.\n'
            'The protocol, IDs and sources were frozen before new trajectories. The student\n'
            'explicitly approved and reported advisor review. No target fitting occurred;\n'
            'this endpoint does not revise the original incremental or operational decisions.\n\n'
        ) + summary[end:]
        summary = summary.replace('by the student, while advisor approval remains pending; no independent preregistration\nis claimed.',
            'by the student; the historical record contains no advisor approval. The extension\nrecords student approval and student-reported advisor review, without independent preregistration.')
    summary_path.write_text(summary, encoding='utf-8')
    typst.compile(str(summary_path), output=str(output / 'research_summary_en.pdf'), root=str(ROOT))
    with pymupdf.open(output / 'research_summary_en.pdf') as pdf:
        if len(pdf) != 2:
            raise ValueError(f'Research summary must be exactly two pages; got {len(pdf)}')
    slides = [
        ('Sinais precoces de corretude', '''*João Felipe Vilas Boas · FGV EMAP*

Prever a trajetória, acrescentar informação e economizar inferência.

Qwen2.5-Coder-1.5B-Instruct · HumanEval+ · MBPP+

Um resultado negativo aprofundado por diagnóstico e análise econômica.'''),
        ('Tarefa, trajetória e ação', '''- AUC agregada pode reconhecer dificuldade do prompt.
- WT-AUC compara corretos e incorretos da mesma tarefa.
- Probabilidades precisam sustentar decisões de aborto.
- Resolução e custo avaliam o sistema sequencial.

Um estado determinístico do prompt empata as tentativas da mesma tarefa.'''),
        ('Desenho e história', '''- Piloto: 960 trajetórias, 32 tarefas.
- HumanEval+: 2.640 tentativas, 132 tarefas.
- MBPP+ congelado: 640 tentativas, 32 tarefas.
- Em 32 tokens: 2.158 candidatos ativos e 92 tarefas mistas.

26 tarefas principais constam de artefatos anteriores. Aprovação do orientador
permanece pendente; não se reivindica pré-registro independente.'''),
        ('Validação por tarefas', '''- Cinco folds externos; tarefa inteira permanece no mesmo fold.
- Seleção interna de C e camada com perda ponderada por tarefa.
- Cross-fitting de toda a seleção antes de Platt.
- Bootstrap pareado de tarefas, condicional aos escores ajustados.

No residual, a seleção e calibração do primeiro estágio também excluem o rótulo pontuado.'''),
        ('Resultado original em 32 tokens', '''#table(columns: 4, inset: 7pt,
 table.header([*Preditor*], [*AUC*], [*WT-AUC*], [*Log loss*]),
 [B], [0,701], [0,628], [0,630], [H], [0,592], [0,568], [0,687],
 [H+B], [0,646], [0,581], [0,668])

ΔWT-AUC = −0,0470 [−0,0994; 0,0056].

Δlog loss = +0,0379 [0,0067; 0,0706].

Os critérios originais de ganho incremental e benefício conjunto não foram satisfeitos.'''),
        ('Leituras alternativas: o que aprendemos?', '''- NLL média e margem preservam parte importante do ranking.
- PCA64: WT-AUC 0,630; log loss 0,632.
- Residual ancorado em B: WT-AUC 0,635; log loss 0,629.
- Intervalos contra B incluem zero; penalização conjunta mais forte não resolveu.

Sensibilidade à parametrização é sustentada; causa única e superioridade geral não são identificadas.'''),
        ('Retenção não é garantia externa', '''#image("../transfer/retention_curves.svg", width: 23cm)

#text(size: 13pt)[Origem OOF, desenvolvimento e alvo congelado são avaliações distintas. Limiares definidos apenas na origem.]'''),
        ('Candidatos descartados e cobertura', '''- H+B no alvo: 276/366 corretos ativos mantidos, 75,41%.
- 90 descartes distribuídos por 12 tarefas; cinco concentram 60.
- Mbpp/166 perde oito sucessos; Mbpp/259 perde seu único sucesso.
- Entre corretos ativos iniciados: 23/48 abortados.

O denominador muda com o caminho de execução. Duas tarefas perdem resolução.'''),
        ('Custo até sucesso', '''Condicionado à tarefa, com tentativas iid:

$ E[C(N)]=E[D] sum_(j=0)^(N-1)(1-p r)^j. $

Para orçamento ilimitado, p>0 e r>0:

$ E[D]/E[T] < r. $

Retenção global, heterogeneidade, cobertura finita e overhead precisam entrar.
Tokens, latência, energia e FLOPs são medidas distintas.'''),
        ('Oracle retrospectivo', '''#image("../economics/oracle_frontier.svg", width: 23cm)

#text(size: 13pt)[Em t=8, N=20: economia idealizada 77,75% em HumanEval+ e 76,25% em MBPP+. Conhece os rótulos; preserva todos os corretos.]'''),
        ('Integridade e limites', '''- Métricas e os três scorers principais reconstituídos.
- 70 unidades de leitura exploratória repetidas com probabilidades idênticas.
- 50 pares / 100 gerações: tokens e estados idênticos entre capturas.
- RTX3050 Laptop 6 GB detectada; tempos técnicos são descritivos.

Um modelo, dois benchmarks, exposição histórica e intervalos condicionais.
Nenhuma causa semântica, garantia externa ou aceleração da política é presumida.'''),
        ('Próximo teste e contribuição', '''Proposta: B e H+B congelados em t=8, 120 novas tarefas MBPP+,
20 tentativas por tarefa e piloto técnico disjunto de 20 tarefas.

Endpoint único: diferença pareada de retenção na mesma população ativa.
Coleta aguarda aprovação do estudante e revisão do orientador.

*Contribuição:* separar sinais, leituras, decisões e oportunidade econômica
em uma avaliação delimitada, explicativa e auditável.'''),
    ]
    preamble = '''#set page(width: 28cm, height: 15.75cm, margin: 1.1cm, numbering: "1")
#set text(font: "Arial", size: 17pt, lang: "pt")
#set par(leading: 0.8em)
#set heading(numbering: none)
#show heading: set text(size: 27pt, fill: rgb("234b68"))
'''
    if independent:
        p = independent['primary']
        effect_pt = effect.replace('.', ',')
        finding_pt = 'satisfeito' if p['directional_hypothesis_supported'] else 'não satisfeito'
        slides[2] = (slides[2][0], slides[2][1].replace(
            'Aprovação do orientador\npermanece pendente;', 'O registro histórico não contém aprovação do orientador;'))
        slides[11] = ('Avaliação independente e contribuição',
            f'''120 novas tarefas MBPP+, 2.400 tentativas principais e 400 técnicas disjuntas.

B e H+B de origem, no mesmo t=8, sem ajuste no alvo.

Retenção H+B−B: {effect_pt}. Critério próprio: {finding_pt}.

Congelamento anterior às trajetórias; aprovação do estudante e revisão do orientador informada por ele.

O novo endpoint não modifica Q2/H3 originais nem identifica uma causa de domain shift.''')
    content = '\n#pagebreak()\n'.join(f'= {title}\n\n{body}\n' for title, body in slides)
    slide_path = output / 'defesa.typ'
    slide_path.write_text(preamble + content, encoding='utf-8')
    typst.compile(str(slide_path), output=str(output / 'defesa.pdf'), root=str(ROOT))
    with pymupdf.open(output / 'defesa.pdf') as pdf:
        if len(pdf) != len(slides):
            raise ValueError(f'Defense layout overflow: expected {len(slides)} slides, got {len(pdf)}')
    inputs = ['reports/excellence/results.json', 'docs/excellence_defense.md', 'scripts/build_excellence_materials.py']
    question_count = len(re.findall(r'^### \d+\.', (ROOT / 'docs/excellence_defense.md').read_text(encoding='utf-8'), re.MULTILINE))
    write_json(output / 'manifest.json', {'created_at': now(), 'research_summary_pages': 2, 'defense_slides': len(slides),
                                         'questions_in_guide': question_count, 'independent_results_claimed': bool(independent),
                                         'inputs': {p: file_hash(ROOT / p) for p in inputs},
                                         'artifacts': {p.relative_to(ROOT).as_posix(): file_hash(p) for p in output.glob('*') if p.suffix in ('.pdf', '.typ')}})
    print(f'Academic materials compiled: research summary 2 pages; defense 12 slides; guide {question_count} questions')


if __name__ == '__main__':
    main()
