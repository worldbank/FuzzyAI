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
def run(input, output, config, id_column, name_column, threshold, validate, validator, resume, verbose):
    """Run deduplication pipeline"""
    
    # Load data
    click.echo(f"Loading data from {input}...")
    try:
        df = load_data(input)
        click.echo(f"Loaded {len(df)} entities")
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)
    
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
@click.option('--output', '-o', default='deduplix_config.yaml', help='Output config file')
def init(output):
    """Create a sample configuration file"""
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


if __name__ == '__main__':
    cli()