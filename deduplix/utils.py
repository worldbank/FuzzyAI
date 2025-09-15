import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to file"""
    
    path = Path(output_path)
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def create_sample_config(output_path: str = "deduplix_config.yaml"):
    """Create a sample configuration file"""
    
    config = {
        'matching': {
            'threshold': 85.0,
            'scorer': 'token_sort_ratio',
            'max_matches_per_entity': 100,
            'n_workers': 4
        },
        'validation': {
            'enabled': True,
            'type': 'llm',
            'rules': {
                'min_score': 90.0
            },
            'llm': {
                'provider': 'openai',
                'model': 'gpt-4',
                'batch_size': 10,
                'n_workers': 4,
                'temperature': 0.1,
                'max_retries': 3
            }
        },
        'pipeline': {
            'checkpoint': True,
            'checkpoint_dir': '.deduplix_checkpoints'
        }
    }
    
    save_config(config, output_path)
    print(f"Sample config created: {output_path}")
    
    return config


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """Load data from various formats"""
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    if path.suffix == '.csv':
        return pd.read_csv(path, **kwargs)
    elif path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(path, **kwargs)
    elif path.suffix == '.parquet':
        return pd.read_parquet(path, **kwargs)
    elif path.suffix == '.json':
        return pd.read_json(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_data(df: pd.DataFrame, file_path: str, **kwargs):
    """Save dataframe to various formats"""
    
    path = Path(file_path)
    
    # Create directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.csv':
        df.to_csv(path, index=False, **kwargs)
    elif path.suffix in ['.xlsx', '.xls']:
        df.to_excel(path, index=False, **kwargs)
    elif path.suffix == '.parquet':
        df.to_parquet(path, **kwargs)
    elif path.suffix == '.json':
        df.to_json(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def analyze_duplicates(result):
    """Analyze deduplication results"""
    
    stats = result.statistics.copy()
    
    # Add more analysis
    if not result.entity_groups.empty:
        group_sizes = result.entity_groups.groupby('group_id').size()
        stats['group_size_distribution'] = {
            'min': int(group_sizes.min()),
            'max': int(group_sizes.max()),
            'mean': float(group_sizes.mean()),
            'median': float(group_sizes.median())
        }
        
        # Find groups by size
        stats['groups_by_size'] = {}
        for size in range(2, min(11, int(group_sizes.max()) + 1)):
            count = (group_sizes == size).sum()
            if count > 0:
                stats['groups_by_size'][f'size_{size}'] = int(count)
    
    return stats