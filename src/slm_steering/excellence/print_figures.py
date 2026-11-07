"""Print layouts of saved results; scientific fitting and statistics are unchanged."""
from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap
import warnings
from unittest.mock import patch

import pandas as pd

from slm_steering.research.protocol import canonical, digest, file_hash, now, write_json


def route_images(root: Path, text: str) -> str:
    manifest = root / 'reports/excellence/print_ready/manifest.json'
    if not manifest.exists():
        return text
    mapping = json.loads(manifest.read_text(encoding='utf-8'))['replacements']
    for original, replacement in mapping.items():
        text = text.replace('../../' + original, '../../' + replacement)
    return text


def render(root: Path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    import matplotlib.pyplot as plt
    from slm_steering.research.reporting import generate_figures
    from slm_steering.research.visualization import additional_figures, transfer_figures
    from slm_steering.excellence.geometry import plot as plot_geometry
    from slm_steering.excellence.transfer import plot as plot_transfer
    from slm_steering.excellence.economics import plot as plot_economics
    from slm_steering.excellence.report import plot_models
    import sys
    sys.path.insert(0, str(root / 'scripts'))
    import research_diagnostics
    import publication_visuals
    output = root / 'reports/excellence/print_ready'
    output.mkdir(exist_ok=True)
    original_save = Figure.savefig
    processed = set()
    layout_rows = []

    def save(fig, filename, *args, **kwargs):
        path = Path(filename)
        if fig not in processed:
            processed.add(fig)
            width, height = fig.get_size_inches()
            # Three density panels become rows; long categorical comparisons keep
            # their vertical room. Only layout objects, never data, are modified.
            vertical_densities = len(fig.axes) == 3 and all(axis.get_xlabel() == 'Probabilidade prevista H+B' for axis in fig.axes)
            if vertical_densities:
                fig.set_size_inches(6.2, 7.15)
                for index, axis in enumerate(fig.axes):
                    axis.set_position([.13, .71 - index * .31, .82, .22])
            else:
                fig.set_size_inches(min(width, 6.2), min(7.15, max(3.6, height)))
            for axis in fig.axes:
                axis.set_xlabel(textwrap.fill(axis.get_xlabel(), 36 if vertical_densities else (24 if len(fig.axes) >= 2 else 36)))
                if vertical_densities and axis is not fig.axes[-1]:
                    axis.set_xlabel('')
                axis.set_ylabel(textwrap.fill(axis.get_ylabel(), 32))
                axis.set_title(textwrap.fill(axis.get_title(), 34))
                legend = axis.get_legend()
                if legend:
                    for item in legend.get_texts():
                        label = item.get_text().replace('HB_PCA16_fixed_layer13_C01', 'H+B PCA16, fixo').replace('B_fixed_C01', 'B, fixo')
                        item.set_text(textwrap.fill(label, 29))
                for item in axis.get_xticklabels():
                    if len(item.get_text()) > 18:
                        item.set_text(textwrap.fill(item.get_text(), 18))
            for item in fig.findobj(Text):
                item.set_fontsize(max(10.5, item.get_fontsize()))
            if len(fig.axes) == 2:
                left_labels = [item.get_text() for item in fig.axes[0].get_yticklabels()]
                right_labels = [item.get_text() for item in fig.axes[1].get_yticklabels()]
                if left_labels == right_labels and any(len(label) > 3 for label in left_labels):
                    fig.axes[1].tick_params(axis='y', labelleft=False)
            if not vertical_densities:
                with warnings.catch_warnings(record=True) as layout_warnings:
                    warnings.simplefilter('always', UserWarning)
                    fig.tight_layout(pad=.7, w_pad=1.2, h_pad=1.2)
                for warning in layout_warnings:
                    print(f'Layout diagnostic {path.stem}: {warning.message}', flush=True)
            layout_rows.append({'stem': path.stem, 'width_inches': float(fig.get_size_inches()[0]),
                                'height_inches': float(fig.get_size_inches()[1]), 'minimum_font_pt': 10.5})
        kwargs['bbox_inches'] = 'tight'
        original_save(fig, filename, *args, **kwargs)
        if path.suffix == '.png':
            original_save(fig, path.with_suffix('.svg'), bbox_inches='tight')

    def load(relative):
        return json.loads((root / relative).read_text(encoding='utf-8'))

    def directory(name):
        value = output / name
        value.mkdir(exist_ok=True)
        return value

    # The signal/policy containers are read from the verified publication snapshot.
    evidence = load('reports/final_review/evidence.json')
    signal, policies = evidence['main']['signal'], evidence['main']['policies']
    predictions = pd.read_csv(root / 'runs/probes/protocol_v0.2/outer_predictions.csv', dtype={'checkpoint': str})
    replacements = {}
    def map_files(original, destination):
        for path in destination.rglob('*'):
            if path.suffix in ('.png', '.svg', '.pdf'):
                relative = path.relative_to(root).as_posix()
                replacements[original + '/' + path.name] = relative
    with patch.object(Figure, 'savefig', save), plt.rc_context({'svg.fonttype': 'none', 'font.family': 'Arial', 'font.size': 10.5}):
        core = directory('core')
        generate_figures(predictions, signal, policies, core)
        additional_figures(predictions, signal, policies, core / 'figures')
        map_files('reports/protocol_v0.2/figures', core / 'figures')
        transfer = directory('frozen_transfer')
        transfer_figures(evidence['transfer'], transfer)
        map_files('reports/transfer_v0.2/figures', transfer)
        geometry = load('reports/excellence/geometry/results.json')
        geo = directory('geometry')
        plot_geometry(geo, pd.DataFrame(geometry['training_spectra']), pd.DataFrame(geometry['learning_curves']))
        map_files('reports/excellence/geometry', geo)
        tr = directory('transfer')
        plot_transfer(tr, pd.read_csv(root / 'reports/excellence/transfer/score_distributions.csv', float_precision='round_trip'),
                      pd.read_csv(root / 'reports/excellence/transfer/retention_threshold_curves.csv'),
                      load('reports/excellence/transfer/results.json')['policies'])
        map_files('reports/excellence/transfer', tr)
        econ = directory('economics')
        plot_economics(econ, pd.DataFrame(load('reports/excellence/economics/results.json')['summaries']))
        map_files('reports/excellence/economics', econ)
        models = directory('models')
        plot_models(models, load('reports/excellence/models/results.json')['summaries'])
        map_files('reports/excellence/figures', models / 'figures')
        extra = directory('supplemental')
        research_diagnostics.plot(extra, load('reports/final_review/supplemental/results.json'))
        map_files('reports/final_review/supplemental/figures', extra / 'figures')
        descriptive = directory('descriptive')
        (descriptive / 'tables').mkdir(exist_ok=True)
        (descriptive / 'legacy_outer_predictions.csv').write_bytes((root / 'reports/final_review/legacy_outer_predictions.csv').read_bytes())
        publication_visuals.render(root, descriptive, evidence)
        map_files('reports/final_review/figures', descriptive / 'figures')
    inputs = ['src/slm_steering/excellence/print_figures.py', 'reports/protocol_v0.2/results.json',
              'runs/probes/protocol_v0.2/outer_predictions.csv', 'reports/transfer_v0.2/results.json',
              'reports/excellence/geometry/results.json', 'reports/excellence/economics/results.json',
              'reports/excellence/models/results.json', 'reports/excellence/transfer/score_distributions.csv',
              'reports/excellence/transfer/retention_threshold_curves.csv']
    inputs = [p for p in inputs if (root / p).exists()]
    extra_result = load('reports/final_review/supplemental/results.json')
    semantic = {key: value for key, value in extra_result.items() if key not in ('created_at', 'inputs', 'artifacts')}
    write_json(output / 'manifest.json', {'created_at': now(), 'scope': 'Saved numerical results, layout only; no fit or new statistical selection',
        'inputs': {p: file_hash(root / p) for p in inputs}, 'supplemental_semantic_sha256': digest(canonical(semantic)),
        'replacements': replacements, 'layouts': layout_rows,
        'artifacts': {p.relative_to(root).as_posix(): file_hash(p) for p in output.rglob('*') if p.suffix in ('.png', '.svg', '.pdf')}})
    print(f'Print layouts: {len(layout_rows)} figures; {len(replacements)} file routes')


def route_generated_chapters(root: Path):
    for path in (root / 'tcc/generated').glob('*.typ'):
        text = path.read_text(encoding='utf-8')
        routed = route_images(root, text)
        # Ensure a print-specific vector replaces a raster counterpart if present.
        routed = re.sub(r'(print_ready/[^"\n]+)\.png', r'\1.svg', routed)
        routed = re.sub(r'(image\("[^"\n]*print_ready/[^"\n]+", width: )(75|90)%', r'\g<1>100%', routed)
        path.write_text(routed, encoding='utf-8')
