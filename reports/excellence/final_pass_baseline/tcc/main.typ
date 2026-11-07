#import "lib.typ": *
#let titulo = "Sinais Latentes Precoces de Corretude em Geração de Código"
#let subtitulo = "Avaliação intratarefa, custo de inferência e transferência de sinais precoces"
#set document(title: titulo, author: "João Felipe Vilas Boas", description: subtitulo)
#set text(font: "Times New Roman", size: 11pt, lang: "pt", region: "BR")
#set page(paper: "a4", margin: (top: 2.7cm, bottom: 2.5cm, left: 3cm, right: 2.4cm))
#set par(justify: true, leading: 0.65em, first-line-indent: 1.15cm)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#set outline(indent: auto)
#set table(inset: 6pt, stroke: 0.35pt + rgb("c5cbd1"))
#show heading: it => block(above: 1.2em, below: 0.7em)[
 #text(hyphenate: false, size: if it.level == 1 {14pt} else {11.5pt}, weight: "bold", it)
]
#show figure.caption: set text(size: 9pt)
#show raw: set text(size: 8.5pt)
#align(center)[
 #text(size: 13pt, weight: "bold")[FUNDAÇÃO GETULIO VARGAS]
 #v(0.2cm)
 ESCOLA DE MATEMÁTICA APLICADA
 #v(2.5cm)
 #text(size: 18pt, weight: "bold", hyphenate: false, titulo)
 #v(0.7cm)
 #text(size: 12pt, subtitulo)
 #v(2cm)
 João Felipe Vilas Boas
 #v(0.5cm)
 Orientador: Prof. Dário Oliveira
 #v(1.2cm)
 #text(size: 10pt)[Manuscrito de pesquisa — avaliação empírica do desenho fixo]
 #v(1fr)
 Rio de Janeiro\ Outubro de 2026
]
#pagebreak()
#set page(numbering: "i")
#counter(page).update(1)
#include "generated/resumo.typ"
#pagebreak()
#outline(title: "Sumário", depth: 2)
#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)
#include "chapters/01_introducao.typ"
#pagebreak()
#include "chapters/02_relacionados.typ"
#pagebreak()
#include "chapters/03_fundamentacao.typ"
#pagebreak()
#include "chapters/04_metodologia.typ"
#pagebreak()
#include "generated/resultados.typ"
#pagebreak()
#include "generated/excellence.typ"
#pagebreak()
#include "chapters/06_discussao.typ"
#pagebreak()
#include "generated/conclusao.typ"
#pagebreak()
#bibliography("referencias.bib", title: "Referências", style: "apa")
#pagebreak()
#include "chapters/08_reprodutibilidade.typ"
