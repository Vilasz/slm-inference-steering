#import "lib.typ": *
#let titulo = "Sinais Latentes Precoces de Corretude em Geração de Código"
#let subtitulo = "Avaliação intratarefa, custo de inferência e robustez entre modelos e decodificadores"
#set document(title: titulo, author: "João Felipe Vilas Boas", description: subtitulo)
#set text(font: "Times New Roman", size: 12pt, lang: "pt", region: "BR")
#set page(paper: "a4", margin: (top: 3cm, bottom: 2cm, left: 3cm, right: 2cm))
#set par(justify: true, leading: 0.84em, first-line-indent: 1.25cm)
#set heading(numbering: "1.1", supplement: [capítulo])
#set math.equation(numbering: "(1)")
#set outline(indent: auto)
#set table(inset: 6pt, stroke: 0.35pt + rgb("c5cbd1"))
#show heading: it => block(above: 1.2em, below: 0.7em)[
 #text(hyphenate: false, size: 12pt, weight: "bold", style: if it.level >= 3 {"italic"} else {"normal"}, if it.level == 1 {upper(it)} else {it})
]
#show heading.where(numbering: none): set align(center)
#show figure.caption: set text(size: 10pt)
#show figure.caption: set par(leading: 0.34em, first-line-indent: 0pt)
#show raw: set text(size: 10pt)
#set figure.caption(position: top)
#show figure: it => block(breakable: false)[#it #block(above: 2pt, below: 6pt)[#text(size: 10pt)[Fonte: elaboração própria a partir dos artefatos e procedimentos identificados na legenda.]]]
#align(center)[
 #text(weight: "bold")[FUNDAÇÃO GETULIO VARGAS]
 #v(0.2cm)
 ESCOLA DE MATEMÁTICA APLICADA
 #v(1.7cm)
 João Felipe Vilas Boas
 #v(2cm)
 #text(weight: "bold", hyphenate: false, upper(titulo))
 #v(0.7cm)
 #text(size: 12pt, subtitulo)
 #v(2cm)
 #v(1fr)
 Rio de Janeiro\ 2026
]
#pagebreak()
#counter(page).update(1)
#align(center)[
 João Felipe Vilas Boas
 #v(3cm)
 #text(weight: "bold", hyphenate: false, upper(titulo))
 #v(0.5cm)
 #subtitulo
 #v(2cm)
]
#align(right, block(width: 55%)[
 #set par(justify: true, leading: 0.34em, first-line-indent: 0pt)
 Trabalho de Conclusão de Curso apresentado à Escola de Matemática Aplicada
 da Fundação Getulio Vargas como requisito para obtenção do grau de bacharel
 em Matemática Aplicada.

 Orientador: Prof. Dário Oliveira.
])
#v(1fr)
#align(center)[Rio de Janeiro\ 2026]
#pagebreak()
#include "generated/resumo.typ"
#pagebreak()
#outline(title: "Sumário", depth: 2)
#pagebreak()
#set page(numbering: (n, ..rest) => text(size: 10pt, str(n)), number-align: right + top)
#include "chapters/01_introducao.typ"
#pagebreak()
#include "chapters/02_relacionados.typ"
#pagebreak()
#include "chapters/03_fundamentacao.typ"
#pagebreak()
#include "chapters/04_metodologia.typ"
#pagebreak()
#include "generated/etapa_qwen.typ"
#pagebreak()
#include "generated/comparacao.typ"
#pagebreak()
#include "chapters/07_discussao_integrada.typ"
#pagebreak()
#include "generated/conclusao.typ"
#pagebreak()
#{
 set par(leading: 0.34em, first-line-indent: 0pt)
 bibliography(("referencias.bib", "referencias_extensao.bib"), title: "Referências", style: "associacao-brasileira-de-normas-tecnicas")
}
#pagebreak()
#counter(heading).update(0)
#set heading(supplement: [apêndice], numbering: (..args) => {
 let values = args.pos()
 if values.len() == 1 { "APÊNDICE " + numbering("A", ..values) + " —" }
 else { numbering("A.1", ..values) }
})
#include "chapters/08_reprodutibilidade.typ"
