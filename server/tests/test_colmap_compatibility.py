from types import SimpleNamespace

from app.pipeline.colmap_common import feature_option_prefixes


def test_feature_option_names_match_cli_generation(monkeypatch):
    import app.pipeline.colmap_common as common
    common.feature_option_prefixes.cache_clear()
    monkeypatch.setattr(common.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='', stderr='--FeatureExtraction.use_gpu arg'))
    assert feature_option_prefixes('new-colmap') == ('FeatureExtraction', 'FeatureMatching')
    monkeypatch.setattr(common.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='--SiftExtraction.use_gpu arg', stderr=''))
    assert feature_option_prefixes('old-colmap') == ('SiftExtraction', 'SiftMatching')
    common.feature_option_prefixes.cache_clear()
