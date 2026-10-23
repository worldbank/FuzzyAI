"""CLI engine commands, exercised with CliRunner on small synthetic data."""
import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from deduplix.cli import cli
from deduplix.engine import PipelineResult, load_pipeline
from tests.data.make_synthetic import make_companies, make_cross

PIPELINE = """
normalize:
  - default_text
blocking:
  - brute_force
compare:
  name: {fn: token_sort_ratio, left: name_norm}
score:
  feature: {name: name}
route:
  threshold: {accept_above: 0.90, reject_below: 0.60}
cluster: components
"""


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def pipeline_file(tmp_path):
    p = tmp_path / "pipeline.yaml"
    p.write_text(PIPELINE, encoding="utf-8")
    return p


@pytest.fixture
def dedup_out(tmp_path, runner, pipeline_file):
    df = make_companies(60, 0.3)
    data = tmp_path / "data.csv"
    df.to_csv(data, index=False)
    out = tmp_path / "out"
    res = runner.invoke(cli, ["run", "--pipeline", str(pipeline_file), "-i", str(data), "-o", str(out)])
    assert res.exit_code == 0, res.output
    return df, out


def test_run_pipeline_saves_engine_and_legacy_files(dedup_out):
    _, out = dedup_out
    for name in ("pairs.parquet", "records.parquet", "meta.json", "entity_groups.csv", "duplicate_pairs.csv"):
        assert (out / name).exists(), name
    assert json.loads((out / "meta.json").read_text())["mode"] == "dedup"


def test_run_without_pipeline_unchanged(tmp_path, runner):
    df = make_companies(40, 0.3)
    data = tmp_path / "d.csv"
    df.to_csv(data, index=False)
    out = tmp_path / "legacy"
    res = runner.invoke(cli, ["run", "-i", str(data), "-o", str(out), "--threshold", "85",
                              "--no-resume"])
    assert res.exit_code == 0, res.output
    assert (out / "entity_groups.csv").exists()
    assert not (out / "pairs.parquet").exists()


def test_link_pipeline(tmp_path, runner, pipeline_file):
    a, b = make_cross(30, 40, 0.5)
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    a.to_csv(pa, index=False)
    b.to_csv(pb, index=False)
    pipe = tmp_path / "link.yaml"
    pipe.write_text(PIPELINE, encoding="utf-8")
    out = tmp_path / "linkout"
    res = runner.invoke(cli, ["link", "--pipeline", str(pipe), "--left", str(pa), "--right", str(pb),
                              "-o", str(out), "--id-column", "company_id", "--name-column", "company_name",
                              "--right-id-column", "entity_id", "--right-name-column", "name"])
    assert res.exit_code == 0, res.output
    assert (out / "right_records.parquet").exists()
    assert (out / "cross_matches.csv").exists() or any(out.glob("*.csv"))
    assert PipelineResult.load(out).mode == "link"


def test_pipeline_extra_sections_are_accepted(tmp_path, runner):
    p = tmp_path / "full.yaml"
    p.write_text(PIPELINE + "security:\n  enable_input_sanitization: true\n"
                 "consolidation:\n  enabled: false\n", encoding="utf-8")
    df = make_companies(30, 0.3)
    data = tmp_path / "d.csv"
    df.to_csv(data, index=False)
    res = runner.invoke(cli, ["run", "--pipeline", str(p), "-i", str(data), "-o", str(tmp_path / "o")])
    assert res.exit_code == 0, res.output


def test_components(runner):
    res = runner.invoke(cli, ["components", "--kind", "blocker"])
    assert res.exit_code == 0
    assert "exact" in res.output and "token_index" in res.output
    assert "weighted" not in res.output
    assert runner.invoke(cli, ["components", "--kind", "nope"]).exit_code != 0


def test_validate_config(tmp_path, runner, pipeline_file):
    ok = runner.invoke(cli, ["validate-config", str(pipeline_file)])
    assert ok.exit_code == 0 and "Config OK" in ok.output
    bad = tmp_path / "bad.yaml"
    bad.write_text("blocking:\n  - no_such_blocker\n", encoding="utf-8")
    res = runner.invoke(cli, ["validate-config", str(bad)])
    assert res.exit_code != 0
    assert runner.invoke(cli, ["validate-config", str(tmp_path / "missing.yaml")]).exit_code != 0


@pytest.mark.parametrize("template", ["basic", "funds", "llm"])
def test_init_template_loads(tmp_path, runner, template):
    out = tmp_path / "pipeline.yaml"
    res = runner.invoke(cli, ["init", "--template", template, "-o", str(out)])
    assert res.exit_code == 0, res.output
    load_pipeline(out, allow_missing_env=True)
    assert runner.invoke(cli, ["validate-config", str(out)]).exit_code == 0


def test_init_without_template_unchanged(tmp_path, runner):
    out = tmp_path / "cfg.yaml"
    assert runner.invoke(cli, ["init", "-o", str(out)]).exit_code == 0
    assert "matching" in out.read_text()


def test_explain(dedup_out, runner):
    _, out = dedup_out
    res = PipelineResult.load(out)
    assert len(res.pairs) > 0
    r = res.pairs.iloc[0]
    id1, id2 = res.records["id"].iloc[int(r["i1"])], res.records["id"].iloc[int(r["i2"])]
    cli_res = runner.invoke(cli, ["explain", str(out), id1, id2])
    assert cli_res.exit_code == 0, cli_res.output
    assert "score:" in cli_res.output and "name1:" in cli_res.output
    assert runner.invoke(cli, ["explain", str(out), "nope", "nada"]).exit_code != 0


def test_evaluate(dedup_out, tmp_path, runner):
    _, out = dedup_out
    res = PipelineResult.load(out)
    m = res.matches
    rows = [{"id1": res.records["id"].iloc[int(a)], "id2": res.records["id"].iloc[int(b)], "label": 1}
            for a, b in zip(m["i1"], m["i2"])]
    gold = tmp_path / "gold.csv"
    pd.DataFrame(rows or [{"id1": "x", "id2": "y", "label": 1}]).to_csv(gold, index=False)
    cli_res = runner.invoke(cli, ["evaluate", str(out), "--gold", str(gold)])
    assert cli_res.exit_code == 0, cli_res.output
    assert "Pair metrics" in cli_res.output and "Cluster metrics" in cli_res.output
    if rows:
        assert "f1: 1.0000" in cli_res.output
