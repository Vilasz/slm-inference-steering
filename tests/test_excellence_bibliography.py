from scripts.fetch_tcc_references import preserve_additional_entries, preserve_source_metadata


def test_reference_refresh_preserves_added_primary_sources():
    existing = '@misc{model,\n title={old}\n}\n\n@article{extra,\n title={additional}\n}\n'
    generated = '@misc{model,\n title={new}\n}\n'
    combined = preserve_additional_entries(existing, generated)
    assert 'title={new}' in combined and 'title={old}' not in combined
    assert '@article{extra,' in combined and 'title={additional}' in combined
    assert combined.count('@misc{model,') == 1
    assert preserve_additional_entries(combined, generated) == combined


def test_reference_refresh_keeps_official_source_metadata_and_updates_version():
    existing = [{'key': 'preprint', 'url': 'https://arxiv.org/abs/2503.01422'},
                {'key': 'official', 'url': 'https://proceedings.mlr.press/v70/guo17a.html'}]
    generated = [{'key': 'preprint', 'url': 'https://arxiv.org/abs/2503.01422v3'}]
    combined = preserve_source_metadata(existing, generated)
    by_key = {row['key']: row['url'] for row in combined}
    assert by_key == {'preprint': generated[0]['url'], 'official': existing[1]['url']}
    assert preserve_source_metadata(combined, generated) == combined
