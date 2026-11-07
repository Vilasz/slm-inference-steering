#import "../lib.typ": *

= Reprodutibilidade e proveniência <chap-reprodutibilidade>

== Separação entre análise e redação

Os resultados empíricos foram concluídos antes da revisão integrada deste manuscrito. A edição reorganiza a argumentação e renderiza tabelas dos artefatos; não gera candidatos, não reajusta probes e não substitui resultados antigos. A versão anterior de 75 páginas, seus fontes e hashes foram arquivados antes da revisão de 7 de novembro de 2026.

O documento atual combina fontes autorais em chapters, templates editoriais em templates e seções numéricas em generated. A renderização integrada lê tanto os resultados históricos quanto os da extensão final. Não depende de carregar um gerador de código ou executar código produzido por modelos.

#raw(".venv/Scripts/python.exe scripts/build_manuscript.py --compile", block: true)

O manifesto editorial identifica hashes das entradas, dos fontes ativos e do PDF. A geração falha se faltar um artefato necessário, se o resultado externo estiver incompleto ou se permanecerem marcadores não resolvidos. O notebook e os relatórios anteriores mantêm sua finalidade e não são apresentados como se já contivessem toda a análise da segunda etapa.

== Artefatos científicos de referência

#grade(caption: [Entradas que permitem localizar os resultados e suas condições.],
 [*Objeto*], [*Arquivo ou diretório*],
 [Primeira etapa e protocolo], [RESEARCH_PROTOCOL.md; reports/protocol_v0.2/results.json.],
 [Transferência inicial], [reports/transfer_v0.2/results.json.],
 [Ablações exploratórias], [reports/excellence/models/results.json.],
 [Validação independente dos gates], [reports/excellence/independent_v1/results.json e relatórios correspondentes.],
 [Protocolo entre modelos], [docs/protocols/FINAL_CROSS_MODEL_EXTENSION_v1.md e sua emenda de recursos.],
 [Reserva e congelamento], [reports/final_extension/task_inventory.json, final_design.json e freeze.json.],
 [Resultados e replay finais], [reports/final_extension/results.json, external_predictions.csv e replay_per_task.csv.],
 [Treinamento e poder], [Manifests dos probes, power_results.json e qwen_training_sensitivity.json.],
 [Contabilidade e integridade], [status.json, historical_hashes.json, model_weight_integrity.json e registros de suspensão.],
)

Os IDs completos, as seeds e os hashes são preservados nos artefatos, evitando listas extensas no corpo do texto. As revisões dos modelos são Qwen 2e1fd397ee46e1388853d2af2c993145b0f1098a e DeepSeek e063262dac8366fc1f28a4da0ff3c50ea66259ca, aplicadas também aos tokenizers. As revisões fixam a condição avaliada; “versão mais recente” não é uma descrição reprodutível.

O ambiente principal utiliza Python 3.12.8, PyTorch 2.12.0+cu126 e Transformers 4.46.3. EvalPlus 0.3.1 roda em Ubuntu/WSL para seus mecanismos POSIX. As versões completas, CUDA, driver, precisão e picos de memória são registrados nos manifestos. A RTX 4060 mencionada na motivação de execução não estava acessível: os números publicados correspondem à RTX 3050 Laptop de 6 GB.

== Cronologia e exposição anterior

O v0.2 foi executado sob autorização do estudante, com snapshot anterior ao banco principal. Seu registro histórico não apresenta aprovação do orientador como concluída, e a execução não é convertida em pré-registro independente. Vinte e seis tarefas principais já apareciam em registros exploratórios anteriores. Essa exposição humana é distinta de vazamento entre folds e permanece uma limitação declarada.

As ablações de parametrização foram posteriores à inspeção dos resultados. A validação de gates em 120 novas tarefas teve autorização e congelamento próprios; o estudante informou revisão do orientador para aquela extensão. Esse relato não é tratado como assinatura ou aprovação retroativa do v0.2.

A segunda etapa foi autorizada em 4 de novembro de 2026. O piloto e os critérios de viabilidade precederam a definição da amostra final; os procedimentos e cem IDs externos foram congelados em 5 de novembro, antes da geração correspondente. O arquivo de trava registra zero resultados externos novos anteriores ao congelamento. Não se reivindica pré-registro independente ou aprovação adicional do orientador para essa etapa.

O inventário considerou as 32 tarefas da transferência, as vinte técnicas da extensão anterior, as 120 da validação independente e controles já utilizados. Exposição local significa presença de resultados anteriores nos artefatos auditados. A disponibilidade pública do benchmark e o histórico de pré-treinamento dos geradores são questões diferentes, que esse inventário não resolve.

== Integridade temporal, funcional e operacional

Os controles incluem alinhamento do estado após o token observado, equivalência prefixo/sequência completa, teacher forcing para probabilidades, reprodução das seeds, ausência de acesso a sufixos nas features, isolamento por tarefa e ajuste de PCA somente no treino. Testes sintéticos conferem empates, tarefas homogêneas, término exatamente no checkpoint, custos de replay e o solver sequencial finito.

A comparação do EvalPlus histórico com o sandbox novo reavaliou 2.640 programas Qwen de origem sem divergência de label. O novo isolamento restringe rede, arquivos e recursos; não altera retrospectivamente a descrição de segurança do worker anterior. Falhas técnicas de preflight foram registradas antes do primeiro candidato válido, sem gerar labels substitutos.

Uma suspensão do Windows por bateria ocorreu de 22h19 de 4 de novembro a 7h59 de 5 de novembro, horário de São Paulo. A interseção comprovada desse intervalo com as sessões foi excluída da contabilidade ativa. O tempo bruto da tentativa afetada permaneceu inalterado; sua correção temporal está em tabela separada. Nenhuma trajetória foi removida ou regenerada por esse motivo.

Tabelas e intervalos publicados são provenientes dos artefatos identificados. A reescrita não muda bootstrap, limiares, multiplicidade ou interpretação dos protocolos. A auditoria científica registra o alcance dos testes: consistência técnica não comprova ausência de toda adaptação humana, de similaridade semântica ou de contaminação no pré-treinamento.

== Assistência computacional e responsabilidade

A preparação utilizou assistência de um agente de programação para implementação, auditoria, organização dos artefatos, renderização e revisão textual. A autoria científica permanece com o pesquisador, que deve conferir a argumentação, as referências, os resultados e a adequação da versão submetida. O registro dessa assistência não substitui a revisão do orientador nem constitui aprovação institucional.
