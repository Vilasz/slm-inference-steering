"""One-time import of the supplied manuscript into editable chapters.

Existing chapter edits are preserved. The archived original is the import source
on subsequent machines; PDF builds use only the chapter files, not the attachment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    archive = ROOT / 'tcc/archive/manuscrito_recebido_2026-10-03.typ'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        if args.source is None:
            raise ValueError('Supply the original manuscript for the first import')
        archive.write_bytes(args.source.read_bytes())
    source = archive.read_text(encoding='utf-8')
    def section(first, following):
        return source[source.index(first):source.index(following)].strip() + '\n'
    def demote(text):
        return re.sub(r'^(=+)', r'\1=', text, flags=re.MULTILINE)
    intro = section('= Introdução', '= Trabalhos relacionados')
    intro = re.sub(r'Este documento substitui.*?(?=\n\n)',
        'A entrega reúne uma reanálise exploratória dos dados históricos, a validação técnica da '
        'coleta revisada e uma implementação do protocolo v0.2. Os resultados precoces '
        'confirmatórios dependem de coleta completa e congelamento científico. Cada capítulo '
        'distingue o que foi observado do que permanece como desenho experimental.', intro, flags=re.DOTALL)
    intro = intro.replace('a versão definitiva', 'o congelamento científico')
    related = section('= Trabalhos relacionados', '= Formulação probabilística')
    related = related.replace('As referências recentes acima foram conferidas em suas páginas primárias para esta versão.',
                               'As referências são identificadas por fontes primárias, com versão explícita quando fixada na bibliografia.')
    related += '''
== Matriz de posicionamento

#set text(size: 9pt)
#table(columns: (1.1fr, 1.4fr, 1.5fr),
 table.header([*Antecedente*], [*Objeto principal*], [*Relação com este estudo*]),
 [Di Cicco], [Corretude da primeira tentativa a partir do prompt], [Motiva separar dificuldade e trajetória],
 [Klassert], [Incerteza introspectiva em respostas e linhas de código], [Motiva o probe simples e limites da generalização],
 [Zhu et al.], [Dificuldade percebida e alocação de inferência], [Comparação no nível da tarefa],
 [Ruan et al.], [Abortos em episódios de agentes], [Retenção certificada difere de meta empírica],
 [Wang et al. / ST-BoN], [Consistência interna entre amostras e truncamento], [Uso de estados internos para economia já possui antecedentes],
 [Este estudo], [Trajetórias da mesma tarefa, observáveis e replay pareado], [Avaliação delimitada; evidência precoce ainda a produzir],
)
#set text(size: 11pt)
'''
    theory = '= Fundamentação matemática e computacional\n\n' + demote(section('= Formulação probabilística', '= Desenho experimental'))
    methods = '= Metodologia\n\n' + '''A implementação revisada segue o protocolo v0.2, de 12 de outubro de 2026.
As regras a seguir definem o experimento principal. Sua implementação e validação
técnica não significam que as avaliações confirmatórias tenham sido executadas.
A reanálise histórica descrita no capítulo de resultados usa um desenho retrospectivo
próprio e não altera os endpoints prospectivos.

''' + demote(section('= Desenho experimental', '= Plano dos experimentos'))
    methods = re.sub(r'O critério de “aumento material de truncamento”.*?(?=\n\n)',
        'O aumento material foi fixado em cinco pontos percentuais acima da menor taxa '
        'combinada de sintaxe inválida, extração vazia ou truncamento entre temperaturas. '
        'Se nenhuma configuração for elegível, a revisão será registrada antes da análise '
        'principal. A coleta parcial não permite escolher uma temperatura.', methods, flags=re.DOTALL)
    methods = methods.replace('Uma implementação com cross-fitting de desenvolvimento registra essa distinção explicitamente.',
        'A implementação v0.2 produz scores por cross-fitting da seleção inteira: '
        'cada partição pontuada é excluída também da escolha de camada e regularização. '
        'O calibrador é ajustado nesses scores; calibrador e limiar continuam sendo ajustes '
        'de desenvolvimento avaliados integralmente em tarefas externas.')
    methods += '''
== Ambiente efetivamente validado

O dispositivo é uma NVIDIA GeForce RTX 3050 6GB Laptop GPU. A geração usa pesos
em bfloat16, projeções lineares calculadas em float32, atenção eager e uma tentativa
por vez, com TF32 desativado. O armazenamento dos vetores selecionados usa float16.
Essa combinação separa precisão dos pesos, computação e armazenamento; descrevê-la
apenas como “inferência FP16” seria inexato. As tentativas técnicas rejeitadas não
forneceram rótulos ao banco. Versões e hashes estão nos manifestos.

== Transferência externa implementada

O fluxo secundário prepara modelos, calibradores e políticas exclusivamente com
HumanEval+, congela o bundle e então permite coletar MBPP+. São planejadas 32
tarefas selecionadas por hash, com 20 tentativas cada. A aplicação não ajusta
parâmetros com rótulos do benchmark alvo. Recalibração ou retreino em MBPP+
constituiriam outro experimento. Esta entrega valida o isolamento em testes,
mas ainda não reporta resultados empíricos de transferência.
'''
    discussion = section('= Discussão e limitações', '= Considerações finais')
    discussion = discussion.replace('= Discussão e limitações\n', '= Discussão e limitações\n\n#include "../generated/discussion_findings.typ"\n', 1)
    chapters = ROOT / 'tcc/chapters'
    chapters.mkdir(exist_ok=True)
    for name, content in [('01_introducao.typ', intro), ('02_relacionados.typ', related),
                          ('03_fundamentacao.typ', theory), ('04_metodologia.typ', methods),
                          ('06_discussao.typ', discussion)]:
        path = chapters / name
        if not path.exists():
            path.write_text('#import "../lib.typ": *\n\n' + content, encoding='utf-8')
    print('Editable chapters prepared; existing authored files preserved')


if __name__ == '__main__':
    main()
