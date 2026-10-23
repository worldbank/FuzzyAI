import click
import pandas as pd
from pathlib import Path
import json
import sys
from .core import DeduplicationPipeline, DeduplicationResult
from .matchers import FuzzyMatcher
from .validators import LLMValidator, RuleBasedValidator
from .utils import load_config, create_sample_config, load_data, save_data, analyze_duplicates


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """deduplix - Entity Deduplication Tool"""
    pass


@cli.command()
@click.option('--input', '-i', required=True, help='Input CSV/Excel file')
@click.option('--output', '-o', required=True, help='Output directory')
@click.option('--config', '-c', help='Configuration file (YAML/JSON)')
@click.option('--id-column', default='id', help='Column name for entity ID')
@click.option('--name-column', default='name', help='Column name for entity name')
@click.option('--threshold', default=80.0, type=float, help='Similarity threshold (0-100)')
@click.option('--validate/--no-validate', default=False, help='Enable validation')
@click.option('--validator', type=click.Choice(['rules', 'llm']), default='rules', help='Validator type')
@click.option('--resume/--no-resume', default=True, help='Resume from checkpoint')
@click.option('--verbose', is_flag=True, help='Verbose output')
@click.option('--pipeline', 'pipeline_path', type=click.Path(), default=None,
              help='Engine pipeline YAML (runs the staged engine instead of the legacy matcher)')
def run(input, output, config, id_column, name_column, threshold, validate, validator, resume, verbose,
        pipeline_path):
    """Run deduplication pipeline"""

    # Load data
    click.echo(f"Loading data from {input}...")
    try:
        df = load_data(input)
        click.echo(f"Loaded {len(df)} entities")
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)

    if pipeline_path:
        _run_engine(pipeline_path, output, df, None, id_column, name_column, None, None, verbose)
        return

    # Load configuration or use defaults
    if config:
        try:
            cfg = load_config(config)
            if verbose:
                click.echo(f"Loaded config from {config}")
        except Exception as e:
            click.echo(f"Error loading config: {e}", err=True)
            sys.exit(1)
    else:
        cfg = {
            'matching': {'threshold': threshold},
            'validation': {'enabled': validate, 'type': validator}
        }
    
    # Initialize matcher
    matcher_cfg = cfg.get('matching', {})
    matcher = FuzzyMatcher(
        threshold=matcher_cfg.get('threshold', threshold),
        scorer=matcher_cfg.get('scorer', 'ratio'),
        max_matches_per_entity=matcher_cfg.get('max_matches_per_entity', 100),
        n_workers=matcher_cfg.get('n_workers', 4)
    )
    
    # Initialize validator if needed
    validator_obj = None
    if validate or cfg.get('validation', {}).get('enabled', False):
        val_cfg = cfg.get('validation', {})
        val_type = val_cfg.get('type', validator)
        
        if val_type == 'llm':
            llm_cfg = val_cfg.get('llm', {})
            validator_obj = LLMValidator(
                provider=llm_cfg.get('provider', 'openai'),
                model=llm_cfg.get('model', 'gpt-4'),
                batch_size=llm_cfg.get('batch_size', 10),
                n_workers=llm_cfg.get('n_workers', 4),
                temperature=llm_cfg.get('temperature', 0.1)
            )
        else:
            rules_cfg = val_cfg.get('rules', {})
            validator_obj = RuleBasedValidator(
                min_score=rules_cfg.get('min_score', 90.0)
            )
    
    # Create pipeline
    pipeline_cfg = cfg.get('pipeline', {})
    pipeline = DeduplicationPipeline(
        matcher=matcher,
        validator=validator_obj,
        checkpoint=pipeline_cfg.get('checkpoint', True),
        checkpoint_dir=pipeline_cfg.get('checkpoint_dir', '.deduplix_checkpoints')
    )
    
    # Run deduplication
    click.echo("\nStarting deduplication pipeline...")
    try:
        result = pipeline.run(
            df,
            id_column=id_column,
            name_column=name_column,
            resume=resume
        )
    except Exception as e:
        click.echo(f"Error during deduplication: {e}", err=True)
        sys.exit(1)
    
    # Save results
    click.echo(f"\nSaving results to {output}...")
    try:
        result.save(output)
        click.echo(f"Results saved successfully")
    except Exception as e:
        click.echo(f"Error saving results: {e}", err=True)
        sys.exit(1)
    
    # Print summary
    click.echo("\n" + "="*50)
    click.echo("DEDUPLICATION COMPLETE")
    click.echo("="*50)
    
    stats = analyze_duplicates(result)
    for key, value in stats.items():
        if isinstance(value, dict):
            click.echo(f"{key}:")
            for k, v in value.items():
                click.echo(f"  {k}: {v}")
        else:
            click.echo(f"{key}: {value}")


@cli.command()
@click.option('--output', '-o', default=None, help='Output config file (default: deduplix_config.yaml, '
                                                    'or pipeline.yaml with --template)')
@click.option('--template', type=click.Choice(['basic', 'funds', 'llm']), default=None,
              help='Write a starter engine pipeline instead of the legacy sample config')
def init(output, template):
    """Create a sample configuration file"""
    if template:
        output = output or 'pipeline.yaml'
        try:
            text = (Path(__file__).parent / 'templates' / f'{template}.yaml').read_text(encoding='utf-8')
            Path(output).write_text(text, encoding='utf-8')
            click.echo(f"Pipeline template '{template}' written: {output}")
        except Exception as e:
            click.echo(f"Error creating config: {e}", err=True)
            sys.exit(1)
        return
    output = output or 'deduplix_config.yaml'
    try:
        create_sample_config(output)
        click.echo(f"Sample configuration created: {output}")
    except Exception as e:
        click.echo(f"Error creating config: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('results_dir')
@click.option('--format', type=click.Choice(['json', 'text']), default='text', help='Output format')
def analyze(results_dir, format):
    """Analyze deduplication results"""
    try:
        result = DeduplicationResult.load(results_dir)
        stats = analyze_duplicates(result)
        
        if format == 'json':
            click.echo(json.dumps(stats, indent=2))
        else:
            for key, value in stats.items():
                if isinstance(value, dict):
                    click.echo(f"\n{key}:")
                    for k, v in value.items():
                        click.echo(f"  {k}: {v}")
                else:
                    click.echo(f"{key}: {value}")
    except Exception as e:
        click.echo(f"Error loading results: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('results_dir')
@click.argument('entity_id')
def show_group(results_dir, entity_id):
    """Show all entities in the same group as the given entity"""
    try:
        result = DeduplicationResult.load(results_dir)
        
        # Convert entity_id to appropriate type
        try:
            entity_id = int(entity_id)
        except ValueError:
            pass  # Keep as string
        
        group_members = result.get_group(entity_id)
        
        if not group_members:
            click.echo(f"Entity {entity_id} not found or has no duplicates")
        else:
            click.echo(f"Group members for entity {entity_id}:")
            for member_id in group_members:
                entity_data = result.entity_groups[
                    result.entity_groups['entity_id'] == member_id
                ]
                if not entity_data.empty:
                    name = entity_data['entity_name'].iloc[0]
                    click.echo(f"  {member_id}: {name}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
@cli.command()
@click.option('--input', '-i', required=True, help='Input CSV/Excel file (original data)')
@click.option('--results', '-r', required=True, help='Deduplication results directory')
@click.option('--output', '-o', required=True, help='Output file for cleaned data')
@click.option('--id-column', default='id', help='Column name for entity ID')
@click.option('--strategy', type=click.Choice(['first', 'last', 'highest_score']), 
              default='first', help='Strategy for keeping duplicates')
@click.option('--verbose', is_flag=True, help='Verbose output')
def remove(input, results, output, id_column, strategy, verbose):
    """Remove duplicates from original data using deduplication results"""
    
    # Load data
    try:
        df = load_data(input)
        if verbose:
            click.echo(f"Loaded {len(df)} entities")
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)
    
    # Load results
    try:
        result = DeduplicationResult.load(results)
        if verbose:
            click.echo(f"Loaded results")
    except Exception as e:
        click.echo(f"Error loading results: {e}", err=True)
        sys.exit(1)
    
    # Remove duplicates
    try:
        cleaned_df = result.remove_duplicates(df, id_column=id_column, keep_strategy=strategy)
        if verbose:
            click.echo(f"Cleaned: {len(cleaned_df)} entities")
    except Exception as e:
        click.echo(f"Error removing duplicates: {e}", err=True)
        sys.exit(1)
    
    # Save
    try:
        save_data(cleaned_df, output)
        click.echo(f"Saved to {output}")
    except Exception as e:
        click.echo(f"Error saving: {e}", err=True)
        sys.exit(1)
    
    # Summary
    summary = result.get_removal_summary(df, cleaned_df)
    click.echo("\n" + "="*40)
    click.echo("REMOVAL COMPLETE")
    click.echo("="*40)
    click.echo(f"Original: {summary['original_count']:,}")
    click.echo(f"Cleaned: {summary['cleaned_count']:,}")
    click.echo(f"Removed: {summary['duplicates_removed']:,}")
    click.echo(f"Rate: {summary['removal_rate']:.1f}%")



# ---------------------------------------------------------------------- engine commands
_EXTRA_SECTIONS = ('consolidation', 'llm_consolidation', 'security')


def _read_pipeline_document(path):
    """Read a pipeline YAML; split off the sections the engine does not own."""
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")
    raw = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    if not isinstance(raw, dict):
        raise ValueError("Pipeline file must contain a mapping")
    extras = {k: raw.pop(k) for k in _EXTRA_SECTIONS if k in raw}
    return raw, extras


def _apply_security(raw, df_list, security):
    """Honour ``security``: rate limit LLM providers, sanitize name text."""
    if not security:
        return
    rpm = security.get('requests_per_minute')
    if security.get('enable_rate_limiting') and rpm:
        for entry in raw.get('validate', []) or []:
            if isinstance(entry, dict):
                for params in entry.values():
                    prov = params.get('provider') if isinstance(params, dict) else None
                    if isinstance(prov, dict) and len(prov) == 1:
                        inner = next(iter(prov.values()))
                        if isinstance(inner, dict):
                            inner.setdefault('requests_per_minute', int(rpm))
    if security.get('enable_input_sanitization'):
        from .security.sanitizer import SecuritySanitizer

        san = SecuritySanitizer(strict_mode=False)
        for df, col in df_list:
            df[col] = [san.sanitize_text(v) if isinstance(v, str) else v for v in df[col]]


def _dataclass_kwargs(cls, section):
    import dataclasses

    names = {f.name for f in dataclasses.fields(cls)}
    kw = {k: v for k, v in section.items() if k in names}
    kw.setdefault('enabled', True)
    return kw


def _run_engine(pipeline_path, output, left, right, id_column, name_column, right_id_column,
                right_name_column, verbose):
    from .engine import load_pipeline

    link = right is not None
    try:
        raw, extras = _read_pipeline_document(pipeline_path)
        sanitize = [(left, name_column)] + ([(right, right_name_column or name_column)] if link else [])
        _apply_security(raw, [(d, c) for d, c in sanitize if c in d.columns], extras.get('security'))
        pipe = load_pipeline(raw)
    except Exception as e:
        click.echo(f"Error loading pipeline: {e}", err=True)
        sys.exit(1)

    click.echo("\nStarting engine pipeline...")
    try:
        if link:
            result = pipe.link(left, right, id_column=id_column, name_column=name_column,
                               right_id_column=right_id_column, right_name_column=right_name_column)
        else:
            result = pipe.dedupe(left, id_column=id_column, name_column=name_column)
    except Exception as e:
        click.echo(f"Error during deduplication: {e}", err=True)
        sys.exit(1)

    click.echo(f"\nSaving results to {output}...")
    try:
        result.save(output)
        legacy = result.to_legacy()
        legacy.save(output)
        if link and result.consolidated is not None:
            result.consolidated.to_csv(Path(output) / 'consolidated.csv', index=False)
        if link:
            _consolidate_legacy(legacy, left, right, extras, output, id_column, right_id_column or id_column,
                                right_name_column or name_column)
        elif extras.get('consolidation') or extras.get('llm_consolidation'):
            click.echo("Note: consolidation sections apply to 'link' only; ignored for dedup", err=True)
    except Exception as e:
        click.echo(f"Error saving results: {e}", err=True)
        sys.exit(1)

    click.echo("\n" + "=" * 50)
    click.echo("DEDUPLICATION COMPLETE" if not link else "LINKAGE COMPLETE")
    click.echo("=" * 50)
    for key in ('n_left', 'n_right', 'n_pairs', 'n_matches'):
        if key in result.stats:
            click.echo(f"{key}: {result.stats[key]}")
    for key, value in legacy.statistics.items():
        click.echo(f"{key}: {value}")


def _consolidate_legacy(legacy, left, right, extras, output, id1, id2, name2):
    from .consolidation import ConsolidationConfig
    from .llm_consolidation import LLMConsolidationConfig

    sec = extras.get('consolidation')
    if isinstance(sec, dict) and sec.get('enabled', True):
        cfg = ConsolidationConfig(**_dataclass_kwargs(ConsolidationConfig, sec))
        out = legacy.consolidate_with_config(left, right, cfg, id_column1=id1, id_column2=id2,
                                             name_column2=name2)
        out.to_csv(Path(output) / 'consolidated_rules.csv', index=False)
        click.echo("Rule-based consolidation written: consolidated_rules.csv")
    sec = extras.get('llm_consolidation')
    if isinstance(sec, dict) and sec.get('enabled', True):
        cfg = LLMConsolidationConfig(**_dataclass_kwargs(LLMConsolidationConfig, sec))
        out = legacy.consolidate_with_llm(left, right, cfg, id_column1=id1, id_column2=id2,
                                          name_column2=name2)
        out.to_csv(Path(output) / 'consolidated_llm.csv', index=False)
        click.echo("LLM consolidation written: consolidated_llm.csv")


@cli.command()
@click.option('--pipeline', 'pipeline_path', required=True, type=click.Path(), help='Engine pipeline YAML')
@click.option('--left', required=True, help='Left CSV/Excel file')
@click.option('--right', required=True, help='Right CSV/Excel file')
@click.option('--output', '-o', required=True, help='Output directory')
@click.option('--id-column', default='id', help='ID column of the left file')
@click.option('--name-column', default='name', help='Name column of the left file')
@click.option('--right-id-column', default=None, help='ID column of the right file (default: same)')
@click.option('--right-name-column', default=None, help='Name column of the right file (default: same)')
@click.option('--verbose', is_flag=True, help='Verbose output')
def link(pipeline_path, left, right, output, id_column, name_column, right_id_column, right_name_column,
         verbose):
    """Match two datasets with an engine pipeline"""
    try:
        df1, df2 = load_data(left), load_data(right)
        click.echo(f"Loaded {len(df1)} left and {len(df2)} right entities")
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)
    _run_engine(pipeline_path, output, df1, df2, id_column, name_column, right_id_column,
                right_name_column, verbose)


@cli.command()
@click.option('--kind', default=None, help='Only list this kind (blocker, comparator, scorer, ...)')
def components(kind):
    """List registered engine components with their parameters"""
    from .engine import list_components
    from .engine.registry import KINDS

    if kind and kind not in KINDS:
        click.echo(f"Unknown kind '{kind}'. Choose from: {', '.join(KINDS)}", err=True)
        sys.exit(1)
    for row in list_components(kind):
        params = ', '.join(row['parameters'])
        click.echo(f"{row['kind']:<12} {row['name']:<22} ({params})")
        if row['doc']:
            click.echo(f"{'':<12} {row['doc']}")


@cli.command('validate-config')
@click.argument('path', type=click.Path())
def validate_config(path):
    """Build a pipeline from PATH without data and report errors"""
    import re

    from .engine import load_pipeline

    try:
        raw, extras = _read_pipeline_document(path)
        text = Path(path).read_text(encoding='utf-8')
        pipe = load_pipeline(raw, allow_missing_env=True)
    except Exception as e:
        click.echo(f"Invalid config: {e}", err=True)
        sys.exit(1)
    import os

    missing = sorted({v for v in re.findall(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', text) if v not in os.environ})
    stages = pipe.stage_configs()
    click.echo(f"Config OK: {path}")
    click.echo(f"  normalize: {len(pipe.normalizers)}  blocking: {len(pipe.blockers)}  "
               f"compare: {len(pipe.comparators)}  validate: {len(pipe.validators)}")
    click.echo(f"  score: {stages['score']}  route: {stages['route']}  cluster: {stages['cluster']}")
    if extras:
        click.echo(f"  extra sections: {', '.join(extras)}")
    if missing:
        click.echo(f"  warning: environment variables not set: {', '.join(missing)}")


def _load_engine_result(path):
    from .engine import PipelineResult

    return PipelineResult.load(path)


def _coerce_id(series, value):
    if pd.api.types.is_numeric_dtype(series):
        try:
            return type(series.iloc[0].item())(value) if len(series) else value
        except (ValueError, TypeError):
            return value
    return value


@cli.command()
@click.argument('results_dir')
@click.argument('id1')
@click.argument('id2')
def explain(results_dir, id1, id2):
    """Print every column for one candidate pair"""
    try:
        res = _load_engine_result(results_dir)
        c = res.columns
        right = res.right_records if res.right_records is not None else res.records
        row = res.explain(_coerce_id(res.records[c.id], id1), _coerce_id(right[c.rid], id2))
    except KeyError as e:
        click.echo(f"Error: {e.args[0] if e.args else e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    for key, value in row.items():
        click.echo(f"{key}: {value}")


@cli.command()
@click.argument('results_dir')
@click.option('--gold', required=True, help='Gold CSV with columns id1, id2, label (1 = same entity)')
def evaluate(results_dir, gold):
    """Print pair, band and cluster metrics against a gold file"""
    from .evaluation.metrics import cluster_metrics, metrics_by_band, pair_metrics

    try:
        res = _load_engine_result(results_dir)
        g = load_data(gold)
        missing = {'id1', 'id2'} - set(g.columns)
        if missing:
            raise ValueError(f"gold file lacks columns {sorted(missing)}")
        c = res.columns
        right = res.right_records if res.right_records is not None else res.records
        pairs = res.pairs
        named = pd.DataFrame({
            'id1': res.records[c.id].to_numpy()[pairs['i1'].to_numpy()],
            'id2': right[c.rid].to_numpy()[pairs['i2'].to_numpy()],
            'score': pairs['score'].to_numpy(),
        })
        pred = named[(pairs['decision'] == 'match').to_numpy()]
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo("Pair metrics")
    for k, v in pair_metrics(pred, g).items():
        click.echo(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    click.echo("\nMetrics by score band")
    click.echo(metrics_by_band(named, g).to_string(index=False))

    if res.mode == 'dedup' and 'cluster_id' in res.records.columns:
        import networkx as nx

        pos = g[g['label'] == 1] if 'label' in g.columns else g
        graph = nx.Graph()
        graph.add_nodes_from(res.records[c.id].tolist())
        graph.add_edges_from(zip(pos['id1'].tolist(), pos['id2'].tolist()))
        gold_labels = {n: i for i, comp in enumerate(nx.connected_components(graph)) for n in comp}
        pred_labels = pd.Series(res.records['cluster_id'].to_numpy(), index=res.records[c.id].to_numpy())
        click.echo("\nCluster metrics")
        for k, v in cluster_metrics(pred_labels, pd.Series(gold_labels)).items():
            click.echo(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    cli()