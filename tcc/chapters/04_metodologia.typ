#import "../lib.typ": *

= Metodologia <chap-metodologia>

== Desenho geral e estatuto das análises

O estudo foi executado em etapas, e não como um único experimento concebido antes de todos os resultados. Esta distinção determina o que cada comparação permite concluir. A primeira etapa corresponde ao protocolo v0.2 com Qwen, probes lineares e avaliação por tarefa em HumanEval+. A segunda compara classes de probes em dois geradores e utiliza um novo conjunto externo comum. Entre elas ocorreram diagnósticos exploratórios e uma validação independente de gates, com hipótese de retenção própria.

#grade(caption: [Papel de cada conjunto na sequência da investigação.],
 [*Conjunto*], [*Uso científico*],
 [Piloto original], [32 tarefas HumanEval+, 960 trajetórias; desenvolvimento da configuração de geração Qwen.],
 [Origem da primeira etapa], [132 tarefas HumanEval+, 2.640 trajetórias Qwen; avaliação OOF por tarefa e seleção restrita ao desenvolvimento.],
 [Transferência inicial], [32 tarefas MBPP+, 640 trajetórias; aplicação do sistema Qwen congelado, sem reajuste no alvo.],
 [Validação independente dos gates], [120 novas tarefas MBPP+, 2.400 trajetórias Qwen; comparação prospectiva de retenção no checkpoint comum de oito tokens.],
 [Origem da segunda etapa], [As mesmas 132 tarefas HumanEval+; reutilização do banco Qwen e coleta de 2.640 trajetórias DeepSeek para desenvolvimento dos oito probes.],
 [Teste externo da segunda etapa], [100 novas tarefas MBPP+ comuns aos geradores; vinte tentativas por tarefa e modelo, totalizando 4.000 gerações.],
)

OOF, de out-of-fold, designa previsões em tarefas excluídas do ajuste do respectivo fold. Não é sinônimo de um novo benchmark. Na segunda etapa, “externo” designa o conjunto MBPP+ reservado para testar procedimentos já congelados na origem. Essa diferença evita misturar o teste agrupado em HumanEval+ com a transferência prospectiva entre benchmarks.

As cinco perguntas da introdução constituem uma síntese editorial. Nos arquivos normativos, Q1, Q2 e Q3 do protocolo original têm seus significados e critérios próprios. A nova hipótese DeepSeek/MLP não substitui o critério original nem transforma as ablações posteriores em testes antecipados. Os registros de autorização e de exposição são preservados no apêndice.

== Geração, corretude e informação disponível

=== Dois modelos e uma especificação funcional comum

Foram utilizados Qwen/Qwen2.5-Coder-1.5B-Instruct e deepseek-ai/deepseek-coder-1.3b-instruct. O primeiro tem 28 blocos e estados de dimensão 1.536; o segundo usa arquitetura Llama com 24 blocos e dimensão 2.048. Pesos e tokenizers foram fixados por revisões imutáveis, reproduzidas no apêndice. Não houve fine-tuning dos geradores nem quantização.

O prompt transmite a mesma especificação de função, com mensagens renderizadas pelo template oficial de cada tokenizer. O renderer histórico mantém uma referência textual a HumanEval também em MBPP; essa imperfeição foi preservada para não alterar somente a condição de teste. No DeepSeek, o template já emite BOS e termina com a marca de resposta; a tokenização evita adicionar um segundo BOS. O token EOT de ID 32021 define seu término. Qwen utiliza suas próprias regras de término congeladas.

A comparação não presume equivalência entre tokens. Trinta e dois tokens podem representar quantidades diferentes de código nos dois geradores, sobretudo quando há comentários ou formatação. Em vez de igualar estados ou entropias de vocabulários distintos, compara-se o incremento de $H$ sobre os observáveis do próprio modelo.

=== Amostragem e independência dos candidatos

O piloto original testou temperaturas 0,6, 0,8 e 1,0 em 32 tarefas separadas das 132 de origem. Uma regra fixada privilegiou a composição mista, sujeita a restrições de validade e truncamento, sem usar AUC de probes. Foram dez tentativas por tarefa e temperatura, somando 960. A configuração selecionada foi temperatura 1,0, top-p 0,95, top-k zero e máximo de 512 novos tokens.

A mesma configuração de amostragem foi mantida na segunda etapa. Cada banco principal contém vinte tentativas por tarefa, inclusive após um sucesso. Isso fornece uma população comum para todos os probes e para o replay; gerar um banco diferente para cada política confundiria diferenças de decisão e de candidatos.

As seeds são derivadas por hash de protocolo, versão do benchmark, tarefa e índice da tentativa, com namespace de modelo próprio na segunda etapa. A reinicialização antes de cada tentativa impede que abortar uma anterior consuma o estado aleatório destinado à seguinte. Não há feedback dos testes para modificar o próximo prompt. Essas condições sustentam o pareamento, mas não certificam independência estatística absoluta ou ausência de variações numéricas entre hardwares.

=== Desfecho funcional e segurança

O desfecho é aprovação nas suítes base e plus do EvalPlus 0.3.1, com HumanEval+ v0.1.10 e MBPP+ v0.2.0. O programa é extraído pela regra de parsing fixada antes dos testes correspondentes. Truncamento é um atributo: um programa terminado pelo limite ainda pode passar. Falha do candidato sob o limite do verificador é um insucesso funcional; falha de comunicação, corrupção ou indisponibilidade da infraestrutura não recebe um rótulo negativo fabricado.

Na execução histórica, EvalPlus utilizou subprocessos WSL, timeouts e reliability_guard. Esses mecanismos não equivalem a isolamento de rede e arquivos. Na segunda etapa, a avaliação foi executada em namespaces WSL/bubblewrap, sem rede, sem home ou disco Windows montados e com runtime e dataset somente para leitura. O limite de memória por processo foi fixado em 3 GiB, em vez do padrão histórico de 4 GiB. A reavaliação dos 2.640 programas Qwen de origem nesse ambiente encontrou zero divergências de label; os rótulos antigos não foram substituídos.

Os limites de tempo e memória fazem parte da definição operacional de corretude. Passar nos testes não constitui prova para todas as entradas, e diferenças de parser ou de ambiente precisam ser consideradas na interpretação externa.

=== Checkpoint, conjunto ativo e causalidade

O checkpoint principal das duas etapas é $t=32$. Nos demais pontos da primeira etapa, $t in {0,8,16,64,"final"}$, as análises são secundárias ou diagnósticas. Uma tentativa entra na decisão precoce somente quando $T_(i j)>t$. Se termina exatamente no checkpoint ou antes dele, segue ao verificador; não recebe uma representação futura fictícia nem é contada como aborto do gate.

O estado é capturado após alimentar o token $t$ ao modelo. Os logits anteriores predizem esse token e não são sua ativação. A instrumentação foi confrontada com processamento do prefixo isolado, sequência completa e geração com cache, dentro das tolerâncias fixadas. Também foram comparados os tokens com e sem captura seletiva e as estatísticas com teacher forcing. Esses testes verificam temporalidade e integridade numérica, sem atribuir causalidade semântica à representação.

O vetor $B$ contém dez features: cinco atributos do prompt — comprimento tokenizado, comprimento da assinatura, presença de import, de exemplos com >>> e de bloco de código — e cinco estatísticas do prefixo: NLL média e última, entropia média e máxima e margem média entre os dois maiores logits. NLL e entropia usam a distribuição amostrada após temperatura e top-p; a margem é calculada antes do top-p, após temperatura. Somente os tokens observados até o checkpoint entram nessas estatísticas. Comprimento final, tempo futuro e resultado do verificador não são preditores.

== Primeira etapa: avaliação Qwen e desenvolvimento exploratório

=== Separação de tarefas e ajuste linear

Das 164 tarefas HumanEval+, 32 foram destinadas ao piloto e 132 ao banco principal. Agrupar por task_id impede que respostas da mesma tarefa apareçam simultaneamente no treino e no teste de um fold. Não remove a exposição humana anterior a tarefas: 26 tarefas principais já apareciam em artefatos históricos, limitação documentada que não foi apagada pela mudança de seeds.

A avaliação utilizou cinco folds externos e três internos agrupados. B, H e H+B receberam regressão logística L2. Para H e H+B, a camada foi selecionada entre profundidades relativas de 25%, 50%, 75% e 100%. O grid de C foi 0,001; 0,01; 0,1; 1 e 10. A seleção minimizou log loss balanceada por tarefa; entre opções dentro de um erro-padrão estimado da melhor, preferiu regularização mais forte e depois camada mais rasa. Essa regra de um erro-padrão é uma heurística de seleção, não um intervalo de confiança.

Imputação, padronização e ajuste foram feitos somente no treino correto. A calibração Platt utilizou scores obtidos por cross-fitting de todo o procedimento de seleção dentro do treino externo. Assim, os labels da tarefa pontuada não participam do modelo, da seleção ou do calibrador que a avalia. As escolhas de desenvolvimento e a comparação final precisam permanecer separadas para controlar o otimismo de seleção @cawley2010.

Os observáveis recebem o mesmo peso total por tarefa: uma tentativa de uma tarefa com menos respostas ativas tem peso maior do que uma tentativa de tarefa com mais respostas ativas. Isso define uma população de tarefas com contribuições iguais, em vez de uma população dominada pelas trajetórias mais longas.

=== Critério original e diagnósticos posteriores

O critério incremental original em 32 tokens exigia diferença positiva de WT-AUC, IC95% pareado inteiramente acima de zero e diferença negativa de log loss. O intervalo da perda também era reportado. Controles de associação por permutação dentro da tarefa eram uma análise distinta, com combinação por folds; não constituíam um teste isolado de informação condicional adicional sobre $B$.

Após conhecer o resultado, foram examinados maior regularização, PCA e um corretor residual sobre a previsão de $B$. Seus ajustes respeitaram as partições, mas a escolha de investigá-los foi posterior. Portanto, seus intervalos são exploratórios e pontuais. Eles motivam a investigação seguinte, sem confirmar a superioridade de uma representação.

=== Transferência e validação dos gates

O transporte inicial aplicou modelos, calibradores e limiares ajustados em HumanEval+ a 32 tarefas MBPP+, com vinte gerações em cada. Não houve retreino nem recalibração com labels alvo. Os gates originalmente escolhidos podiam usar checkpoints distintos; essa avaliação examinava os procedimentos transportados completos.

Uma validação independente posterior reservou vinte tarefas técnicas e 120 tarefas principais sem resultados locais anteriores. Sua hipótese própria comparou a retenção de corretos ativos dos gates Qwen B e H+B, ambos em oito tokens, com limiares congelados na origem. A retenção é uma razão de contagens de candidatos, e não uma média de AUCs ou a proporção de tarefas resolvidas. Os intervalos dessa hipótese utilizaram 10.000 reamostragens de tarefas inteiras, recalculando a razão em cada réplica. Seus resultados não alteram o contraste original em 32 tokens.

== Segunda etapa: comparação entre modelos e classes de probes

=== Piloto DeepSeek e inventário externo

Doze tarefas HumanEval+ do desenvolvimento original foram fixadas por hash, separadas das 132 de origem, para um piloto DeepSeek de vinte tentativas por tarefa. Os critérios para prosseguir incluíram pelo menos quatro tarefas mistas, taxa agregada de acertos de pelo menos 5%, truncamento de no máximo 25%, erro de parsing de no máximo 15% e aprovação dos controles numéricos e funcionais. Qwen foi medido em uma tentativa nas mesmas doze tarefas para dimensionamento técnico; esse conjunto não compõe o teste externo.

A auditoria MBPP+ considerou 378 tarefas, 173 IDs já expostos em trajetórias ou tabelas locais e uma triagem lexical de variantes. Depois das exclusões, 189 tarefas permaneceram elegíveis. Similaridade lexical não é garantia de independência semântica completa. A ordenação foi determinada por hash antes das novas gerações; não se substituíram tarefas por seus resultados.

A hierarquia de tamanho prevista foi 150, 140, 120, 100 ou 80 tarefas por modelo, com vinte tentativas. A projeção baseada no piloto, com contingência de 50%, selecionou cem tarefas para respeitar aproximadamente 60 horas de sessão ativa. A redução foi uma aplicação da regra de recursos, não uma reação à significância. Os mesmos cem IDs foram usados pelos dois modelos.

A análise de planejamento simulou efeitos conhecidos de +0,02, +0,03, +0,05 e +0,075, usando a dispersão histórica por tarefa e cenários de frequência mista. Seus intervalos normais são aproximações de planejamento, distintos do bootstrap final. As curvas e a possibilidade de baixo poder para efeitos pequenos foram registradas antes do teste; poder calculado a partir do efeito externo observado não foi usado como validação.

=== Features e comparação fatorial

Na segunda etapa, H foi fixado na saída do bloco de profundidade relativa de 75%: índice 20 no Qwen e 17 no DeepSeek, em base zero. A escolha é comum em profundidade relativa, mas não alinha semanticamente as coordenadas. O estado passou por padronização e PCA64 com whitening estimados somente no treino. O vetor B manteve a definição anterior, computado pelo tokenizer do respectivo gerador.

#grade(caption: [Quatro famílias centrais aplicadas separadamente a cada gerador.],
 [*Conjunto de features*], [*Classe do probe*],
 [$B$], [Regressão logística L2.],
 [$H+B$], [Regressão logística L2 com o mesmo grid de regularização.],
 [$B$], [MLP de uma camada oculta ReLU, largura 16 ou 32.],
 [$H+B$], [O mesmo desenho de MLP, acrescentando os 64 componentes de H.],
)

Cada contraste H+B−B é calculado dentro da mesma classe. Na comparação entre logística e MLP, a arquitetura do gerador, o checkpoint e a representação latente permanecem fixos. O grid e a validação do baseline são próprios e correspondentes; B não é mantido deliberadamente subajustado para favorecer H+B.

=== Desenvolvimento na origem, calibração e congelamento

Os probes foram desenvolvidos nas mesmas 132 tarefas HumanEval+ por modelo. O banco Qwen válido foi reutilizado; para DeepSeek, coletaram-se vinte tentativas por tarefa. Por ordenação hash, 26 tarefas foram reservadas para calibração e limiar. As outras 106 alimentaram validação agrupada de três folds para selecionar hiperparâmetros pela média da log loss balanceada por tarefa. As 26 tarefas não participaram dessa seleção.

A logística utilizou C em 0,001; 0,01; 0,1 e 1. O MLP utilizou uma camada ReLU de 16 ou 32 unidades, dropout 0,1, AdamW com taxa de aprendizado 0,001 e weight decay 0,001. O treinamento foi em CPU, full batch, com BCE ponderada por tarefa e limite de duzentas épocas.

Dentro de cada treino interno, uma reserva agrupada de 20% forneceu early stopping, com paciência de vinte épocas e melhora mínima de 0,00001. Ela não incluiu o fold pontuado. Após escolher o número de épocas, o modelo foi reajustado no treino interno completo. O ajuste final nas 106 tarefas utilizou a mediana das épocas escolhidas nos folds. As seeds 20261104, 20261105 e 20261106 foram todas mantidas: o ensemble é a média de probabilidades, sem seleção do membro mais favorável.

A calibração Platt e o limiar foram ajustados somente nas 26 tarefas reservadas. O limiar busca preservar pelo menos 97% dos corretos ativos de calibração. Esse número é uma meta de desenvolvimento, não uma garantia de retenção externa. Os modelos, transformações, calibradores, limiares, código e IDs foram congelados em 5 de novembro de 2026, às 19h29 UTC, antes de qualquer resultado externo novo. Nenhuma decisão de ajuste utilizou os labels dessas cem tarefas.

=== Hipótese primária e comparações secundárias

A hipótese primária foi o incremento de WT-AUC de H+B/MLP sobre B/MLP no DeepSeek aos 32 tokens. O IC95% pareado por tarefa tem interpretação bilateral: inteiramente acima de zero sustenta ganho; inteiramente abaixo sustenta deterioração; incluindo zero não demonstra diferença. Não foi fixada uma margem de equivalência que permitisse concluir igualdade prática por ausência de significância.

DeepSeek linear, ambos os contrastes Qwen, a interação $Gamma$, as diferenças probabilísticas, a calibração e o replay foram secundários. Seus intervalos são pontuais, sem controle conjunto de multiplicidade. A interpretação não promove um resultado secundário ao lugar do contraste primário quando este é inconclusivo.

=== Estabilidade do treinamento

Além dos três membros fixos do MLP, oito análises de sensibilidade na origem Qwen repetiram seleção, pré-processamento, ajuste e calibração completos. Um conjunto fixo de 26 tarefas foi reservado exclusivamente para auditar estabilidade; cada repetição utilizou 84 das 106 tarefas restantes, sem reposição. Não se utilizaram labels MBPP novos.

Essa subamostragem evita que cópias de uma mesma tarefa atravessem folds, mas não é um bootstrap de treinamento. Oito conjuntos sobrepostos, um conjunto de auditoria pequeno e um treino menor não fornecem uma estimativa sem viés da variância do procedimento final. A análise quantifica sensibilidade empírica e não converte os IC finais condicionais em intervalos de toda a seleção.

== Métricas, denominadores e inferência

A WT-AUC é a média sem ponderação das AUCs das tarefas mistas no conjunto ativo. Cada tarefa informativa contribui uma vez; todos os pares positivo-negativo de uma mesma tarefa continuam dependentes. As tarefas homogêneas não recebem AUC artificial de 0,5. Elas permanecem nas perdas probabilísticas quando possuem candidatos ativos e nas métricas operacionais de toda a população.

AUC agregada, average precision, log loss e Brier foram calculados com pesos iguais por tarefa contribuinte. Log loss mede a penalização probabilística de acertos e erros, dando peso elevado a erros confiantes; Brier mede erro quadrático das probabilidades. Calibração foi examinada em dez bins fixos. Essas medidas não substituem a ordenação intratarefa.

O bootstrap reamostra tarefas inteiras, com o mesmo índice para todos os métodos comparados. A primeira etapa utilizou 2.000 réplicas e seed 20260921; a validação independente de gates utilizou 10.000 e seed 20261017; a segunda etapa utilizou 10.000 e seed 20261104. Na WT-AUC, contribuem as tarefas mistas; nas perdas, as tarefas com candidatos ativos; no custo e na resolução, todas as tarefas selecionadas. Comparações entre geradores mantêm o cluster da tarefa. O contraste de incrementos entre modelos é reportado separadamente nas tarefas mistas comuns, pois muda a população em relação aos contrastes primários.

Os intervalos condicionam nos probes ajustados e no conjunto observado. Não incorporam integralmente a variabilidade de refazer seleção, calibração e treinamento. Também não certificam ausência de perdas raras quando nenhuma foi observada: se todas as diferenças de resolução forem zero, o bootstrap empírico pode devolver um intervalo degenerado. Essa limitação será considerada na discussão operacional.

== Replay e contabilidade do custo

As políticas são avaliadas nos mesmos candidatos, na ordem original de tentativa, para $N in {1,5,10,20}$. O baseline fixo gera todos os candidatos; o verificador encerra ao primeiro sucesso. Os gates interrompem candidatos ativos cujo score é inferior ao limiar e continuam a busca até um sucesso mantido ou o esgotamento. O oracle contrafactual conhece os labels e aborta apenas incorretos ativos. Ele não é implementável em uma aplicação real.

Uma tentativa abortada custa t tokens de completion; uma mantida custa seu comprimento integral. O replay soma somente tentativas efetivamente iniciadas. Tarefas não resolvidas contribuem com seus custos. Tokens agregados por tarefa resolvida dividem o custo de todas as tarefas pela quantidade resolvida, e não apenas pelo custo das que tiveram sucesso.

A retenção entre todos os corretos ativos do banco responde a uma pergunta diferente da taxa de aborto entre corretos ativos iniciados. Um sucesso potencial descartado depois de uma solução anterior não teria sido iniciado pelo replay; o primeiro denominador o inclui, e o segundo não. Ambos são identificados nas tabelas.

O protocolo original condicionou uma afirmação conjunta de utilidade ao suporte incremental, à não inferioridade de resolução com margem de três pontos percentuais, à economia contra verificador e B e à validação online de replay. A segunda etapa não herdou uma nova hipótese confirmatória operacional: seus resultados de custo são secundários, com limiares da origem e sem concluir redução de latência a partir de tokens.

== Condições computacionais e integridade

A GPU efetivamente utilizada foi uma RTX 3050 Laptop de 6 GB. Os pesos permaneceram em BF16, as projeções densas foram calculadas em FP32, as ativações armazenadas em float16, com atenção eager, batch um e TF32 desligado. Precisão de armazenamento, computação e pesos são condições distintas e foram mantidas durante cada comparação.

Os candidatos foram gravados de forma atômica e a retomada verificou identidades, seeds e hashes. Uma suspensão do Windows por bateria atravessou uma sessão da segunda etapa. O intervalo de 9h40 foi comprovado pelo log do sistema e excluído somente da contabilidade de sessão ativa, preservando tempos brutos e todos os candidatos. O total ativo inclui carga e verificação; não é perfil de kernels CUDA. Essa emenda técnica ocorreu antes da avaliação externa e não alterou amostra ou critério estatístico.

Extração, score do probe, geração e verificação têm custos próprios. Medições em lote de score na CPU não estimam automaticamente overhead online por tentativa. A análise mantém separados tokens de completion, tempo de execução, custo de preparação da pesquisa e desempenho de uma política já treinada. Os artefatos que permitem reconstruir essas condições são descritos no apêndice.
