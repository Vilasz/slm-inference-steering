# Sinais latentes precoces de corretude em geração de código

Trabalho de Conclusão de Curso de **João Felipe Vilas Boas**, FGV EMAp, 2026.
O estudo avalia Qwen2.5-Coder-1.5B-Instruct e DeepSeek-Coder-1.3B-Instruct:
os estados internos de um prefixo curto ajudam a distinguir tentativas corretas
da mesma tarefa, além de observáveis desse prefixo? Essa previsão melhora uma
política de interrupção e seu custo?

A conclusão é delimitada: **capacidade preditiva dos estados não implica
incremento aproveitável por um probe nem benefício econômico confiável**.
Os contrastes primários não demonstraram ganho incremental. Isso não prova
inexistência de sinal latente ou equivalência entre os procedimentos.

## Comece por aqui

| Arquivo | Conteúdo |
|---|---|
| [Manuscrito final — PDF](tcc/build/manuscrito.pdf) | Texto completo, tabelas, figuras, referências e apêndice |
| [Revisão final](reports/finalization/FINAL_REVIEW.md) | Correções, testes, limites e dez perguntas de banca |
| [Auditoria estatística](reports/finalization/STATISTICAL_AUDIT.md) | Reconstrução dos números a partir de scores e registros salvos |
| [Matemática e bibliografia](reports/finalization/MATHEMATICAL_AND_BIBLIOGRAPHIC_AUDIT.md) | Equações, solver e fontes prioritárias verificadas |
| [Reprodução final](docs/FINAL_REPRODUCTION.md) | Ambiente mínimo e comandos em CPU |
| [O que publicar no GitHub](docs/FINAL_PUBLICATION.md) | Pasta de entrega, seleção de arquivos e manifesto |

## Resultados principais

A primeira etapa usa Qwen, 132 tarefas HumanEval+ e 2.640 tentativas, com
validação aninhada por tarefa. Em 32 tokens, ΔWT-AUC(H+B−B) = **−0,047**,
IC95% **[−0,099; 0,006]**, e Δlog loss = **+0,038**, IC95% **[0,007; 0,071]**.
O critério incremental e o benefício operacional conjunto não foram demonstrados.
PCA e correção residual são diagnósticos exploratórios posteriores.

A validação independente anterior, com 120 tarefas MBPP+ novas e gates
congelados de oito tokens, encontrou retenção de corretos ativos de **82,72%**
para H+B e **99,08%** para B: diferença de **−16,36 pontos percentuais**.
É uma conclusão específica de transporte daqueles gates.

A segunda etapa compara B e H+B dentro de cada classe — regressão logística
ou MLP — nos dois geradores. O desenvolvimento usa HumanEval+; o teste externo
contém as mesmas cem tarefas MBPP+ por modelo e vinte tentativas por tarefa,
totalizando quatro mil novas gerações.

| Gerador | Classe | ΔWT-AUC H+B−B | IC95% | Tarefas mistas | Estatuto |
|---|---|---:|---|---:|---|
| DeepSeek | MLP | −0,0042 | [−0,0648; 0,0548] | 58 | Primário |
| DeepSeek | Logística | −0,0038 | [−0,0668; 0,0554] | 58 | Secundário |
| Qwen | MLP | −0,0074 | [−0,0944; 0,0766] | 35 | Secundário |
| Qwen | Logística | −0,0552 | [−0,1347; 0,0206] | 35 | Secundário |

O resultado primário DeepSeek/MLP é inconclusivo. A interação secundária positiva
no Qwen, Γ = 0,0478, IC95% [0,0178; 0,0810], representa redução da desvantagem
linear; não demonstra superioridade absoluta de H+B/MLP. Os IC são condicionais
aos probes treinados e usam tarefas como unidades de reamostragem.

Algumas políticas economizam tokens sem perda observada de tarefas no banco
externo. Isso não certifica retenção futura nem aceleração online. No DeepSeek,
B/MLP não aborta nenhuma tentativa iniciada; H+B/MLP apresenta economia pequena.
As diferenças de população ativa, calibração, treinamento e número de tarefas
mistas limitam a generalização. [Números completos e denominadores](reports/finalization/STATISTICAL_AUDIT.md).

## Organização da entrega final

```text
README.md                         apresentação da edição final
LICENSE                           licença existente do projeto
requirements-final.txt            dependências de reconstrução em CPU
tcc/
  main.typ                        entrada do manuscrito
  chapters/                       capítulos autorais ativos
  templates/                      prosa com valores preenchidos pelo builder
  generated/                      seções numéricas da edição final
  referencias*.bib                bibliografia
  build/manuscrito.pdf             único manuscrito PDF da entrega
scripts/                          construção, auditoria e fontes preservados
src/slm_steering/                  módulos usados e dependências locais
docs/                             reprodução, publicação e protocolos
reports/finalization/             revisão final e reconstruções verificáveis
reports/final_extension/           resultados e manifests da segunda etapa
reports/                          evidências anteriores exigidas pelo texto
runs/                             mínimo exigido pelos hashes históricos
tests/                            testes selecionados
PUBLICATION_MANIFEST.json          inventário e SHA256 da cópia exportada
```

O workspace de pesquisa mantém experimentos e versões anteriores. A cópia de
publicação contém somente os arquivos selecionados para a edição final e sua
proveniência. O histórico mínimo dos manifests continua incluído; os pesos dos
geradores, as ativações extensas, os backups e os notebooks parciais ficam fora.
Os bundles `.joblib` incluídos são probes estatísticos leves, não os SLMs.

## Reprodução em CPU

Use Python 3.12. Na raiz do projeto, em PowerShell:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-final.txt
.venv/Scripts/python.exe scripts/finalization_audit.py
.venv/Scripts/python.exe scripts/build_manuscript.py --compile
.venv/Scripts/python.exe scripts/audit_integrated_manuscript.py
.venv/Scripts/python.exe -m pytest -q tests/test_final_extension_resources.py
```

Em Linux/macOS, use `python3.12 -m venv .venv` e `.venv/bin/python`.
A compilação com a diagramação original requer **Times New Roman** instalada.
A fonte não é redistribuída. O PDF fornecido já está compilado.

Esses comandos reconstruem métricas, IC, poder, replay e manuscrito; não iniciam
coleta, carregamento de geradores ou treinamento. O CSV compacto das quatro mil
tentativas e os scores salvos bastam para a auditoria externa. A reconstrução
não reexecuta os testes funcionais dos programas nem estima a incerteza completa
do treinamento. Os detalhes e a distinção em relação ao ambiente histórico
estão no [guia de reprodução](docs/FINAL_REPRODUCTION.md).

Para editar o texto, altere os capítulos ativos ou os templates e reconstrua.
Evite editar somente `tcc/generated/`, que será preenchido novamente.

## O que colocar no GitHub

**Publique o conteúdo de `dist/github_final/`, incluindo os arquivos ocultos.**
Prepare a pasta a partir do workspace de pesquisa:

```powershell
.venv/Scripts/python.exe scripts/export_final_repository.py
.venv/Scripts/python.exe scripts/export_final_repository.py --verify
```

O exportador produz uma cópia curada e `PUBLICATION_MANIFEST.json`, com a lista
exata dos arquivos, tamanho e SHA256. Ele preserva o workspace, o índice do Git
e as alterações anteriores. Não executa commit ou push.

Copie esse conteúdo para a pasta destinada ao repositório público. Não é
necessário subir o workspace inteiro. A cópia possui `.gitignore` próprio e
preserva os fins de linha por `.gitattributes`, para que os hashes científicos
continuem válidos depois do Git. Para atualizar uma exportação já existente,
use `--refresh`; uma pasta que já contenha `.git/` é recusada.

O [guia de publicação](docs/FINAL_PUBLICATION.md) explica a seleção e as exclusões.
O inventário gerado é a referência exata do que enviar.

## Protocolos e proveniência

- [Protocolo v0.2](RESEARCH_PROTOCOL.md): desenho original Qwen.
- [EXCELLENCE_EXTENSION_v1](docs/protocols/EXCELLENCE_EXTENSION_v1.md): validação independente dos gates anteriores.
- [FINAL_CROSS_MODEL_EXTENSION_v1](docs/protocols/FINAL_CROSS_MODEL_EXTENSION_v1.md): comparação prospectiva entre geradores e classes de probes.
- [Congelamento final](reports/final_extension/freeze.json): fontes, bundles e insumos protegidos.
- [Verificação numérica final](reports/finalization/statistical_verification.json): resultados reconstruídos e hashes dos insumos.

Os protocolos e registros históricos mantêm suas datas e estados originais.
A revisão final não altera hipóteses, resultados negativos ou decisões tomadas
antes da avaliação externa. A autoria científica e a escolha da versão a ser
submetida permanecem com o pesquisador, em diálogo com seu orientador.