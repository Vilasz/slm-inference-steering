# Emenda técnica de contabilidade de recursos — 05/11/2026

O Windows confirmou suspensão/hibernação por bateria de
2026-11-05T01:19:03.2301052Z até 2026-11-05T10:59:40.1302982Z
(22h19 de 04/11 até 7h59 de 05/11, America/Sao_Paulo).
O evento Power-Troubleshooter 1, registro 80627, foi exportado com XML original
e hash em `reports/final_extension/windows_power_events_20261105.json`.

A sessão de coleta e uma tentativa atravessaram essa suspensão. Os relógios
de parede incluíram aproximadamente 9h40min sem execução de GPU. A correção
remove apenas a interseção comprovada dessas suspensões com as sessões, unindo
intervalos sobrepostos para não duplicar o desconto. Não são descontados
carregamento de modelo, verificação funcional ou espera operacional comum.
Não se denomina essa medida perfil de kernels CUDA.

Os manifests e todos os registros por candidato, incluindo tempos brutos,
permanecem intactos. Ajustes de tempo de candidatos afetados são exportados
separadamente; nenhuma tentativa é removida, regenerada ou tem label alterado.
O limite continua 60 horas de sessão ativa, sem ampliar amostra ou trocar IDs,
modelos, seeds, hiperparâmetros, hipótese ou critério de parada estatística.
O desenho externo permanece fixado em cem tarefas por modelo.

Esta correção é anterior ao congelamento confirmatório e a qualquer geração
externa. Seu instante exato, evidência e hashes constam em
`resource_accounting_adjustments.json`; esses arquivos entram na trava final.
O histórico v0.2 não é corrigido nem reinterpretado por esta emenda.
