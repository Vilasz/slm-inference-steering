#let caixa(titulo, body) = block(width: 100%, inset: 10pt,
  fill: rgb("f0f4f7"), stroke: (left: 2pt + rgb("245c80")),
  above: 0.8em, below: 0.8em)[
  #set par(first-line-indent: 0pt)
  *#titulo* #linebreak() #body
]
#let grade(..cells) = {
  set text(size: 9pt)
  set par(first-line-indent: 0pt, justify: false)
  let c = cells.pos()
  table(columns: (1.05fr, 2.3fr), table.header(..c.slice(0, 2)), ..c.slice(2))
}
#let pendente(body) = caixa("Estado da evidência", body)
