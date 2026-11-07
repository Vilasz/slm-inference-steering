# Revisão final do TCC

Missão iniciada em 07/11/2026, 21:41:15 UTC, com limite de 180 minutos.
O controle de tempo e o encerramento estão em `run.json`. A edição recebida
já tinha 58 páginas e integrava as duas etapas; a referência a 75 páginas na
missão descrevia a apresentação anterior. A revisão atua sobre os capítulos
efetivamente incluídos em `tcc/main.typ`.

## 1. Arquivos alterados e entregues

| Grupo | Arquivos ou diretórios |
|---|---|
| Entrada pública | `README.md`, `.gitignore`, `requirements-final.txt` |
| Prosa e referências | `tcc/templates/resumo.typ.tpl`, `conclusao.typ.tpl`, `comparacao.typ.tpl`; `tcc/chapters/02_relacionados.typ`, `03_fundamentacao.typ`; `tcc/referencias.bib`, `tcc/referencias_extensao.bib`; `tcc/README.md` |
| Construção e auditoria editorial | `scripts/render_integrated_manuscript.py`, `scripts/audit_integrated_manuscript.py` |
| Ferramentas novas | `scripts/finalization_audit.py`, `scripts/export_final_repository.py`, `scripts/verify_publication_roundtrip.py` |
| Documentação nova | `docs/FINAL_REPRODUCTION.md`, `docs/FINAL_PUBLICATION.md` |
| Evidências novas | `reports/finalization/CRITICAL_ISSUES.md`, `STATISTICAL_AUDIT.md`, `MATHEMATICAL_AND_BIBLIOGRAPHIC_AUDIT.md`, este relatório, JSONs de verificação e CSVs derivados |
| Saídas reconstruídas | Seções ativas em `tcc/generated/`, figuras da edição integrada, manifest editorial, validação e `tcc/build/manuscrito.pdf` |

O snapshot inicial de 32 arquivos editoriais e o estado/diff do Git foram
guardados localmente em `reports/finalization/before/`. Protocolos, resultados
científicos e bancos originais não foram editados. A lista exata da cópia de
publicação fica em `dist/github_final/PUBLICATION_MANIFEST.json`.

## 2. Correções científicas e matemáticas

- Resumo e abstract identificam o resultado original negativo, os diagnósticos
  exploratórios, a validação independente de 120 tarefas e o teste prospectivo
  DeepSeek/MLP. A tradução conserva os estimandos e a conclusão inconclusiva.
- A conclusão responde explicitamente a Q1–Q5 da introdução. Ela separa
  previsibilidade, incremento, capacidade/modelo, transporte e decisão/custo.
- A interpretação de Γ permanece secundária: reduzir a desvantagem linear no
  Qwen não demonstra incremento absoluto de H+B/MLP.
- O texto quantifica a atuação efetiva dos gates DeepSeek: B/MLP aborta 0/752
  tentativas iniciadas; H+B/MLP, 7/754. Economizar poucos tokens não estabelece
  benefício após contabilizar captura, score e verificação.
- O custo agregado por tarefa resolvida é apresentado com seu denominador,
  incluindo no numerador o custo das tarefas sem sucesso.
- Bellman explicita o débito de geração, aquisição e verificação no orçamento.
  O manuscrito mantém k como novas tentativas além da atual e usa V_k também
  na regra simplificada. V_0=λ e o caso V=0 são explicitados.
- A condição pV ≥ c exige fallback idêntico após aborto e fracasso. Não é
  apresentada como regra ótima geral sob orçamento estrito.
- Russell e Wefald (1991) foram acrescentados à bibliografia ativa, com dados
  conferidos. As quatro fontes prioritárias estão documentadas na auditoria
  matemática e bibliográfica.
- A inspeção visual identificou “In:” sem evento em referências de anais.
  Acrescentou-se o campo `eventtitle` a quatro entradas, preservando seu tipo
  bibliográfico e o estilo ABNT. Um controle novo impede a volta dessa saída.

## 3. Resultados reproduzidos

A reconstrução independente por contagem de pares e replay reproduziu, com
tolerância absoluta de 1e−11:

- Os contrastes H2 originais Qwen: ΔWT-AUC −0,0470, IC95% [−0,0994; 0,0056],
  e Δlog loss +0,0379, IC95% [0,0067; 0,0706], usando suas 2.000 réplicas e seed.
- A validação independente dos gates: Δretenção −0,1636, IC95%
  [−0,2402; −0,0943], em 120 tarefas, com 10.000 réplicas e seed histórica.
- Os quatro contrastes externos de WT-AUC e respectivos IC; os denominadores
  são 58 tarefas mistas DeepSeek e 35 Qwen.
- Γ DeepSeek −0,0003, IC95% [−0,0313; 0,0323], e Qwen +0,0478,
  IC95% [0,0178; 0,0810], calculados sobre diferenças conjuntas por tarefa.
- O contraste entre modelos nas 28 tarefas mistas comuns.
- As oito linhas de métricas, os contrastes de log loss/Brier e os 80 bins
  descritivos de calibração.
- Os 540 cenários de poder e 135 células do cenário histórico por cálculo
  independente; tamanhos analíticos para 80% de poder e erro Monte Carlo.
- As 5.600 linhas de replay e 56 sínteses, com resolução, tokens, retenção,
  falsos abortos, custos por resolvida e intervalos.
- Os 216 casos finitos do solver sintético e 27 comparações do limiar.

As evidências completas estão em `statistical_verification.json`,
`STATISTICAL_AUDIT.md` e `reproduction/`. Os arquivos científicos protegidos
foram conferidos antes e depois da auditoria.

## 4. O que não foi reexecutado

Não foram repetidos geração, treinamento científico, seleção de hiperparâmetros ou
avaliação funcional dos programas. A tarefa exigia preservar os resultados e
priorizar verificações leves. Os dados e scores existentes bastaram para os
cálculos acima. A autorização posterior de GPU foi registrada, mas nenhuma
etapa desta revisão exigiu seu uso.

Os testes unitários incluem ajustes mínimos de MLP em dados sintéticos
(32 amostras, cinco épocas), usados apenas para conferir determinismo. Eles
não treinam novamente os probes científicos nem entram nos resultados do TCC.

Não se reestimaram a distribuição de treinamento completa, os testes de
permutação históricos, latência online ou contaminação de pré-treinamento.
O relatório não trata a conferência de um hash como nova validação funcional
de um programa ou como releitura integral de uma referência.

## 5. Inconsistências resolvidas

| Triagem | Resolução |
|---|---|
| C01 | Métricas, IC, Γ, poder e replay reconstruídos; sem divergência numérica relevante |
| C02 | Validação independente identificada no resumo e abstract |
| C03 | README principal reescrito para a edição com dois geradores e comandos atuais |
| C04 | Seleção explícita da cópia pública, incluindo apenas o histórico mínimo exigido pelos hashes |
| C05 | Russell/Wefald acrescentados; quatro fontes prioritárias conferidas |
| C06 | Conclusão vinculada explicitamente às cinco perguntas |
| C07 | Tabela de probes permanece inteira com cabeçalho; escopo de dois SLMs preservado |
| C08 | Orçamento e fronteiras esclarecidos; convenção de k unificada no manuscrito |
| C09 | SVGs ativos incorporados ao manifest; configuração pública preserva bytes através do Git |
| C10 | Eventos dos anais aparecem na bibliografia; prefixos vazios eliminados |

## 6. Limitações que permanecem abertas

Os IC condicionam nos probes ajustados e são imprecisos para efeitos pequenos.
A amostra de cem tarefas não prometia alta detecção de ganhos de 0,02–0,05.
As oito perturbações de treino não são uma estimativa sem viés da variância do
procedimento completo. Tarefas mistas e conjuntos ativos diferem entre modelos.

Vinte e seis tarefas originais já tinham exposição exploratória local;
separar folds por tarefa não elimina decisões humanas informadas por esse
histórico. A reserva nova foi triada contra a exposição local conhecida.

Dois SLMs e funções curtas não representam toda a geração de software. O
baseline B é uma compressão específica do prefixo, não o melhor preditor
textual possível. A triagem local não elimina contaminação de pré-treinamento.
Testes finitos não provam correção universal dos programas.

O replay não mede aceleração online e não certifica ausência de perdas raras.
Os resultados secundários têm IC pontuais, sem controle de multiplicidade.
Nenhuma dessas limitações foi tratada como motivo para selecionar outra
hipótese ou alterar retrospectivamente a amostra.

## 7. Testes e verificações

- 231 testes específicos passaram: `tests/test_final_extension.py` e
  `tests/test_final_extension_resources.py`, executados em CPU com downloads
  desabilitados. Registro: `tests.xml`.
- Auditoria numérica independente com bootstrap, simulação, replay e
  invariantes de identidade/denominador; sem refit.
- Ruff nos scripts de publicação e auditoria; conferência de whitespace do Git.
- Compilação sem avisos de citações ou referências cruzadas.
- Auditoria geométrica de largura e altura do conteúdo do PDF.
- Inspeção visual de resumo, abstract, equações, tabela de probes, contrastes,
  calibração, custos, conclusão e referências.
- Uma cópia extraída por Git passou novamente pela auditoria estatística,
  build, auditoria do PDF e três testes leves em um ambiente separado de
  27 pacotes, sem PyTorch/Transformers/datasets. Os hashes sobreviveram ao Git
  com conversão automática de fins de linha habilitada.
- `publication_verification.json` registra o snapshot datado testado e os
  hashes dos scripts; `dependency_verification.json` identifica o ambiente
  mínimo. O manifest da entrega lista seus arquivos finais, incluindo os
  próprios registros de verificação acrescentados após o teste.
- O índice e HEAD do Git permaneceram intactos, assim como 22 diferenças
  preexistentes de arquivos rastreados fora de README/.gitignore; os 32 backups
  editoriais permanecem íntegros. Registro: `git_preservation.json`.

## 8. PDF final e entrega

O PDF final é `tcc/build/manuscrito.pdf`, com 58 páginas. Mantém A4, Times New
Roman 12 pt, margens e estilo bibliográfico ABNT. Tabelas possuem numeração,
cabeçalhos e fontes; as legendas novas identificam métricas e populações.
O manifest e a validação estão em `reports/manuscript_revision_20261107/`.

A entrega para o GitHub é o conteúdo de `dist/github_final/`. O inventário
registra cada arquivo e SHA256; o exportador não faz commit ou push e não
altera o índice do repositório de pesquisa. O perfil mínimo reconstrói os
cálculos em CPU, sem pesos dos geradores ou ativações extensas.

## 9. Pontos para revisão do orientador

Cabe ao orientador avaliar a adequação do recorte e da contribuição aos
critérios do curso, a suficiência das limitações expostas, a interpretação dos
endpoints e a clareza da defesa matemática. Também cabe ao autor conferir
nomes, referências, regras institucionais e a declaração de assistência
computacional antes da submissão. Nenhuma aprovação nova foi criada ou
atribuída ao orientador por esta revisão.

## 10. Avaliação crítica

A versão é coerente com os cálculos reproduzidos e explicita seus estimandos,
denominadores, cronologia e limites. Seu mérito é uma avaliação controlada
que distingue sinal preditivo, incremento acessível e decisão econômica. O
resultado primário inconclusivo continua sendo uma limitação real da evidência.
A redação e a reprodução melhoram a defesa dessas conclusões; não aumentam o
poder da amostra nem transformam a programação dinâmica em contribuição inédita.

## Perguntas mais difíceis que uma banca poderia fazer

1. **Como sustentar uma conclusão com pouco poder?** O estudo não sustenta
   equivalência: o IC primário [−0,0648; 0,0548] ainda admite efeitos relevantes.
   O planejamento já estimava apenas 36,1% de detecção para +0,05 em cem tarefas.
   A conclusão é ausência de ganho demonstrado nesse desenho.
2. **Dois modelos permitem generalizar sobre arquitetura?** Não isolam esse
   efeito. Tokenizers, treinamento e comprimentos também diferem; há 58 contra
   35 tarefas mistas. A análise pareada nas 28 comuns restringe o estimando,
   sem transformar a comparação em experimento causal de arquitetura.
3. **O resultado negativo prova que H não contém informação?** Não. O contraste
   original −0,047 compara estimadores específicos com B. Capacidade, amostra
   e regularização limitam a leitura; uma desvantagem de um probe não prova
   informação mútua condicional nula.
4. **Γ positivo não confirma que o MLP resolveu o problema?** Γ Qwen = +0,0478
   compara dois incrementos. O incremento absoluto MLP é −0,0074, IC95%
   [−0,0944; 0,0766]. A interação reduz uma desvantagem e é secundária.
5. **O que significa informação latente se H é função do prefixo?** Com modelo
   determinístico, H não cria informação além do prompt e prefixo completos.
   Ele pode tornar padrões acessíveis a uma classe limitada de probes em relação
   aos dez resumos B; essa é a comparação efetivamente testada.
6. **O oracle é um limite inferior universal?** Não. Ele conhece labels finais,
   mantém todos os corretos e usa a mesma ordem/checkpoint. Seu limite vale
   nessa classe. Políticas que descartam sucessos podem seguir outros caminhos,
   com custos ou cobertura diferentes.
7. **Qual o benefício prático demonstrado?** No DeepSeek, H+B/MLP reduz tokens
   médios de 1.260,38 para 1.243,65, com 71 tarefas resolvidas e um correto
   iniciado abortado. É uma economia secundária pequena, que não inclui todo
   o overhead e não demonstra redução de latência ou confiabilidade futura.
8. **Por que usar pV ≥ c se há orçamento finito?** É uma redução com fallback
   idêntico e unidades de custo comuns. Se completar muda orçamento ou crença,
   os fallbacks diferem e a recursão completa é necessária. Os gates de recall
   não foram ajustados como solução dessa programação dinâmica.
9. **As hipóteses foram escolhidas depois de olhar os resultados?** PCA e
   correção residual são explicitamente exploratórios. A validação dos gates
   e a extensão entre modelos têm protocolos e congelamentos posteriores,
   anteriores a seus respectivos testes. A narrativa integrada mantém essa
   cronologia e os endpoints separados.
10. **Onde está a originalidade matemática ou científica?** Não se reivindica
    um novo teorema de Bellman ou da soma geométrica. A contribuição é a
    avaliação conjunta de discriminação intratarefa, valor incremental e
    decisões de interrupção, com controles correspondentes e proveniência.
