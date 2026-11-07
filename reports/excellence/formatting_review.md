# Revisão de formatação e escopo institucional

Fontes oficiais consultadas em 17/10/2026:

- [Manual de normalização do Sistema de Bibliotecas FGV, 2025](https://www18.fgv.br/mailing/2025/SB/normalizacao_trabalhos_academicos/), páginas 13–17, 20–22, 24, 27 e 30. HTMLs e imagem da tabela tipográfica estão preservados em `primary_sources/fgv_manual_2025_page*`.
- [Comunicado oficial da biblioteca, 17/02/2025](https://biblioteca.fgv.br/noticia/sistema-de-bibliotecas-fgv-disponibiliza-novos-materiais-para-padronizacao-de-trabalhos): apresenta a revisão apoiada nas normas identificadas no comunicado.
- [Página oficial da graduação em Matemática Aplicada](https://emap.fgv.br/curso/matematica-aplicada) e seu guia do aluno 2026.2, arquivado em `primary_sources/fgv_emap_guia_2026_2.pdf`.

| Item do manual geral | Implementação verificável |
|---|---|
| Papel A4; superior/esquerda 3 cm, inferior/direita 2 cm | `tcc/main.typ`, configuração de página |
| Times New Roman ou Arial; corpo 12 pt | Times New Roman 12 pt no manuscrito |
| Fonte menor para legendas, fontes e paginação | 10 pt; tabelas em 10 pt com espaçamento simples para leitura |
| Texto justificado; recuo de primeira linha 1,25 cm | Configuração global de parágrafo |
| Espaçamento de corpo correspondente a 1,5; exceções simples | Leading 0,84 em no corpo Times 12 pt, calibrado no PDF para baselines de aproximadamente 18 pt; referências, natureza e tabelas com leading 0,34 em para espaçamento simples. A medida geométrica foi conferida; não se declara equivalência integral à composição pelo Word. |
| Hierarquia de títulos | Primeiros níveis em negrito, primário em maiúsculas; terceiro em negrito e itálico; títulos sem numeração centralizados |
| Capa e folha de rosto | Capa reordenada e folha de rosto com natureza, curso e orientador; sem data de defesa ou aprovação inventada |
| Contagem desde a folha de rosto; pré-textuais sem número impresso; arábicos no texto | Capa excluída; contador iniciado na folha de rosto, sem reinício na introdução; número no alto à direita |
| Resumos e palavras-chave | Português e inglês em páginas separadas, cada resumo em parágrafo único; palavras-chave e keywords |
| Identificação de tabelas/figuras e fonte | Legenda acima e indicação da elaboração própria/artefatos abaixo, mantidas no mesmo bloco; seis quadros textuais com títulos e numeração própria; figuras vetoriais com fontes ampliadas |
| Referências e citações | Estilo ABNT do Typst, testado com citação autor-data e datas de acesso; metadados e versões conferidos nas fontes primárias |

O guia EMAP 2026.2 consultado é geral e não estabelece regras específicas de
diagramação de TCC. Por isso, a aderência implementada é ao manual geral da
biblioteca consultado; não se declara homologação pela EMAP nem conformidade
integral com todas as revisões de normas em 2026.

Permanecem externos: confirmação pelo curso da versão/modelo exigido, necessidade
de ficha catalográfica para este grau, emissão da ficha pela biblioteca, folha
de aprovação após defesa real, nomes/assinaturas da banca e depósito no sistema
institucional. Esses elementos não foram fabricados. A revisão visual de todos
os PDFs finais será registrada separadamente, vinculada aos hashes efetivos.
