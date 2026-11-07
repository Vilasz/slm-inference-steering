# Execução de desenvolvimento — 2026-10-03

Status: **piloto parcial; nenhuma conclusão confirmatória**.

| Item | Medida |
|---|---:|
| Trajetórias verificadas | 39 / 960 |
| Tarefas representadas | 13 / 32 |
| Temperaturas | 0,6; 0,8; 1,0 |
| Tokens de continuação | 2693 |
| Tempo de geração do banco | 427.16 s |
| Tempo de verificação do banco | 4.53 s |
| Velocidade efetiva de geração | 6.30 tokens/s |
| Pico reservado observado | 4598 MiB |
| Maior erro L2 relativo no teste causal | 0.00026230 |

GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU. Python 3.12.8, PyTorch
2.12.0+cu126, CUDA 12.6. Pesos bfloat16, projeções lineares
float32 temporárias, backend eager, batch 1, TF32 desativado. A configuração
está em `configs/protocol_v0.1.yaml`; o manifesto copiado está em
[pilot_manifest.json](pilot_manifest.json).

O teste causal passou para 0/8/16/32/64/final, tanto no forward completo
quanto na geração com KV cache. O maior erro observado ficou abaixo da
tolerância predefinida de 1%. A repetição da primeira tentativa com a mesma
semente reproduziu os IDs dos tokens. Sessões concluídas: 2.
Validação adicional do hash de um prefixo do banco antes/depois da retomada:
aprovada; consulte
`resume_validation.json` quando presente.

EvalPlus 0.3.1 executou as suítes base e plus integrais no Ubuntu via WSL;
controles positivo e negativo passaram. O dataset HumanEval+ é v0.1.10.
Os hashes do benchmark, tokens, texto, código e ativações foram auditados.
Nenhuma falha de inicialização/numericalidade recebeu rótulo científico.

No preflight, float16 gerou logits não finitos e bfloat16 nativo excedeu a
tolerância causal. Os registros dessas tentativas técnicas estão em
`runs/diagnostics/`. A configuração validada foi escolhida por validade
numérica antes da coleta de trajetórias rotuladas, sem comparar desempenho
de probes, taxa de acerto ou economia de uma política.

A extrapolação mecânica da duração média deste pequeno lote dá cerca de
**3.0 h para o piloto completo** e **8.1 h para
o banco definitivo**, somente geração/verificação. Essas estimativas são
provisórias: comprimentos, prefill, temperatura, carga e condições térmicas
podem mudar. Elas excluem os testes iniciais, carregamento do modelo, análise
estatística e MBPP+. O limite por sessão controla tempo de execução, não reduz
o número de tentativas nem o teto de 512 tokens.

A temperatura continua não selecionada. Nenhum probe foi aplicado ao piloto.
Complete as 960 trajetórias, prepare o registro de sampling e obtenha o
congelamento previsto pelo protocolo antes do banco definitivo. MBPP+ permanece
uma análise futura; seu subset deverá ser definido por hash de task_id antes
de inspecionar desfechos.
