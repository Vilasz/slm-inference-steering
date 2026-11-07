# Interrupção precoce como decisão sequencial delimitada

Esta é uma aplicação de programação dinâmica, não uma reivindicação de teorema
inédito. O controle de computação por seu valor esperado antecede este estudo:
[Russell e Wefald, Principles of metareasoning (1991)](https://doi.org/10.1016/0004-3702(91)90015-C),
[Callaway et al., Learning to select computations (2018)](https://arxiv.org/abs/1711.06892).
Uma conexão específica com amostragem de LLMs é
[Wan et al., BEACON (2025)](https://arxiv.org/abs/2510.15945), que decide quando
encerrar a busca por respostas completas. A aplicação abaixo distingue continuar
um prefixo, abandonar uma tentativa e iniciar outra; não afirma novidade dessa
estrutura geral. As fontes foram consultadas em 04/11/2026.

## Objetivo, unidade e estado

Para uma tarefa, minimizar

\[
\mathbb E_\pi\left[\sum_{j<\tau}(d_j+g_j+v_j)
 +\lambda\,\mathbf1\{\text{encerra sem solução verificada}\}\right],
\qquad \sum_{j<\tau}(d_j+g_j+v_j)\le b_0.
\]

Os custos de geração \(d\), gate \(g\), verificação \(v\), orçamento \(b_0\)
e penalidade \(\lambda\) usam a mesma unidade: por exemplo milissegundos medidos
ou custo monetário. Tokens podem substituir a unidade apenas num modelo que
converta gate/verificador para equivalentes explicitamente; ignorá-los define
outro objetivo. A penalidade expressa a preferência entre custo e insucesso.
Custos já pagos são afundados e não reaparecem no custo futuro.

O estado de uma tentativa ativa é \((k,b,h,z)\): \(k\) tentativas novas ainda
disponíveis **além da atual**, \(b\) orçamento restante, \(h\) prefixo observado
(incluindo checkpoint e observáveis causais), e \(z\) crença sobre a tarefa
atualizada pelas tentativas anteriores. A informação latente H é determinística
dado o prompt, os tokens e o modelo: pode auxiliar um decodificador restrito,
mas não cria informação adicional à sequência integral sob uma descrição
determinística ideal. Um score escalar estimado não é automaticamente um estado
de Markov suficiente para a distribuição de sucesso e custo restante.

Formalmente, com modelo e instrumentação M fixos, H_t=f_M(P,X_1:t). Logo,
\(I(Y;H_t\mid P,X_{1:t},M)=0\), pela mensurabilidade de H em relação ao prefixo
completo. Isso não implica \(I(Y;H_t\mid B_t,M)=0\): os dez observáveis B são
uma compressão muito mais restrita. O experimento compara decodificabilidade
incremental por classes limitadas de probes; uma diferença negativa de AUC
entre estimadores ajustados não prova nulidade da informação condicional.
Essa identidade conhecida ajuda a delimitar o sentido de “informação latente”,
sem reivindicar uma nova propriedade teórica dos modelos.

## Recursão geral

Denote \(J_k(b,z)\) o menor custo esperado sem tentativa ativa e \(F_k(b,h,z)\)
o custo no checkpoint. Uma tentativa nova custa \(a\) até o primeiro checkpoint,
incluindo adquirir o primeiro sinal sob o calendário fixado, e gera um estado
inicial \(H_0\). Com custos determinísticos por transição e
ações inviáveis removidas quando seu custo excede b:

\[
J_k(b,z)=\min\{\lambda,\ a+\mathbb E[F_{k-1}(b-a,H_0,z)]\},\quad k\ge1,
\qquad J_0(b,z)=\lambda.
\]

No checkpoint **após obter o score**, as ações são encerrar sem solução,
abortar a atual (voltando a \(J_k\)) ou continuar até o próximo checkpoint/evento
terminal. Escreva \(d(h)\) para o custo de geração dessa transição e \(g'\) para
o custo de adquirir o sinal no próximo checkpoint, se houver. O custo do gate
atual já foi pago e não aparece apenas no ramo de continuar:

\[
F_k(b,h,z)=\min\{\lambda,\ J_k(b,z),\
 d(h)+\mathbb E[\Phi_k(b-d(h),H',Z')]\}.
\]

\(\Phi\) vale \(g'+F_k(b-g',H',Z')\) se houver outro checkpoint e a política
adquirir esse novo sinal; num término, paga-se o custo de verificação v,
e o custo futuro é zero se correto ou \(J_k(b-v,Z')\) se
incorreto. O custo v também precisa caber no orçamento. Custos aleatórios
exigem explicitar sua distribuição e remover ações que possam violar um
orçamento estrito, ou definir deliberadamente uma restrição apenas em esperança.
Não basta substituir custos aleatórios por médias numa restrição estrita.
Na transição a um checkpoint posterior, se b não comportar o sinal obrigatório,
encerra-se sem solução. Esta recursão supõe aquisição nos checkpoints fixados.
Se adquirir o gate também for uma decisão, acrescente um estado anterior à
observação com uma ação de pagar g e transição para o estado posterior ao score.
Não se cobra o gate já observado somente das trajetórias que o gate manda continuar.

As condições de fronteira incluem: sucesso verificado tem valor zero;
encerramento sem solução custa \(\lambda\); sem recurso suficiente, só ações
gratuitas admissíveis permanecem; sem novas tentativas, ainda pode ser ótimo
continuar a atual. O horizonte é finito: há um teto de tentativas e tokens por
tentativa. A regra ótima dentro do modelo escolhe o argumento mínimo da
recursão. Sua validade requer transições conhecidas ou crenças suficientes e
custos aditivos na unidade especificada.

As trajetórias podem ser independentes **condicionalmente à tarefa** e às seeds
sob amostragem fixada. Marginalmente, compartilham dificuldade; fracassos prévios
podem mudar z. O código sintético faz a hipótese mais forte iid com p conhecido,
mantendo z constante. Um verificador perfeito é outra hipótese do modelo, não
uma afirmação de correção universal das suites finitas EvalPlus.

## Derivação do caso de um único checkpoint

Suponha que: continuar completa a atual sem novos checkpoints; o custo futuro
esperado c_t inclui geração, verificação e eventual overhead ainda não pago;
sucesso tem probabilidade p_t; o fallback após falhar é exatamente o mesmo
estado que após abortar, com custo ótimo esperado V_{k-1}. Aqui k conta a
tentativa atual entre as k oportunidades, ao contrário da convenção de F acima.
Essa mudança de indexação explica o subscrito k−1.

\[
Q_{\rm continuar}=c_t+(1-p_t)V_{k-1},\qquad
Q_{\rm abortar}=V_{k-1}.
\]

Subtraindo os dois custos, continuar é preferível ou empatado se e somente se

\[
c_t-p_tV_{k-1}\le0
\quad\Longleftrightarrow\quad p_tV_{k-1}\ge c_t.
\]

p_t é adimensional; c_t e V_{k−1} são custos esperados na mesma unidade.
V_{k−1} **não é** a probabilidade de sucesso da próxima tentativa nem um prêmio
arbitrário. É o custo futuro ótimo, incluindo penalidade por insucesso, ao
desistir da atual. Para V>0 o limiar é c_t/V; para V=0 só um custo c_t=0 pode
empatar. Se c_t>V, nenhuma probabilidade admissível torna continuar melhor.
Quando não há novas tentativas, V_0=λ, e a condição vira p_tλ≥c_t.

A hipótese de fallback idêntico é decisiva. Com orçamento estrito, completar
consome recurso e muda o fallback. Então a comparação correta é

\[
c_t+(1-p_t)J_{k-1}(b-c_t,z_{\rm falha})
 \le J_{k-1}(b,z_{\rm aborto}),
\]

e não se reduz em geral a p_tV≥c_t. A correlação entre duração, sucesso e
informação de fracasso pode exigir ainda uma esperança conjunta.

## Relação com os gates e validação sintética

O gate empírico usa um limiar fixo escolhido na origem para retenção de 97% dos
corretos ativos. Ele aproxima uma política homogênea, sem estimar c_t nem J;
não é a política ótima de Bellman e essa meta não certifica retenção externa.
Um limiar ótimo pode depender de tarefa, prefixo, custo restante, k e b. AUC
avalia ordenação; o custo decisório depende da calibração e dos erros no limiar.
Overhead pago antes da decisão é afundado, mas integra a comparação entre usar
ou não usar um gate. Ganho de tokens e ganho de latência são objetivos distintos.

`final_extension/decision.py` resolve exatamente uma versão finita com custos
inteiros e probabilidades conhecidas. Os testes comparam a recursão a enumeração
de todas as ações em estados pequenos, verificam fronteiras, orçamento e a
desigualdade no caso simplificado. A validade desses testes é interna ao modelo
sintético e não demonstra benefício econômico dos probes nos benchmarks.
