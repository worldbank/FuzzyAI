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


def remove_duplicates_after_deduplication(
    original_df: pd.DataFrame,
    dedup_result,
    id_column: str = 'id',
    keep_strategy: str = 'first'
) -> pd.DataFrame:
    """Remove duplicate rows based on deduplication results"""

    entity_groups = dedup_result.entity_groups.copy()
    entities_to_keep = []

    # Keep singletons (group_id = 0)
    singletons = entity_groups[entity_groups['group_id'] == 0]['entity_id'].tolist()
    entities_to_keep.extend(singletons)

    # Process duplicate groups
    duplicate_groups = entity_groups[entity_groups['group_id'] > 0].groupby('group_id')

    for group_id, group_df in duplicate_groups:
        group_entities = group_df['entity_id'].tolist()

        if keep_strategy == 'first':
            original_indices = []
            for entity_id in group_entities:
                idx = original_df[original_df[id_column] == entity_id].index
                if len(idx) > 0:
                    original_indices.append((entity_id, idx[0]))

            if original_indices:
                original_indices.sort(key=lambda x: x[1])
                entities_to_keep.append(original_indices[0][0])

        elif keep_strategy == 'last':
            original_indices = []
            for entity_id in group_entities:
                idx = original_df[original_df[id_column] == entity_id].index
                if len(idx) > 0:
                    original_indices.append((entity_id, idx[-1]))

            if original_indices:
                original_indices.sort(key=lambda x: x[1])
                entities_to_keep.append(original_indices[-1][0])

        elif keep_strategy == 'highest_score':
            if not dedup_result.duplicate_pairs.empty:
                entity_scores = {}
                for entity_id in group_entities:
                    pairs_as_id1 = dedup_result.duplicate_pairs[
                        dedup_result.duplicate_pairs['id1'] == entity_id
                    ]
                    pairs_as_id2 = dedup_result.duplicate_pairs[
                        dedup_result.duplicate_pairs['id2'] == entity_id
                    ]

                    scores = []
                    scores.extend(pairs_as_id1['similarity_score'].tolist())
                    scores.extend(pairs_as_id2['similarity_score'].tolist())

                    entity_scores[entity_id] = sum(scores) / len(scores) if scores else 0

                if entity_scores:
                    best_entity = max(entity_scores.keys(), key=lambda x: entity_scores[x])
                    entities_to_keep.append(best_entity)
                else:
                    entities_to_keep.append(group_entities[0])
            else:
                entities_to_keep.append(group_entities[0])
        else:
            entities_to_keep.append(group_entities[0])

    return original_df[original_df[id_column].isin(entities_to_keep)].copy()


def simple_remove_duplicates(original_df: pd.DataFrame, dedup_result, id_column: str = 'id'):
    """Simple removal - keep first entity from each duplicate group"""
    entity_groups = dedup_result.entity_groups
    entities_to_keep = []

    # Keep singletons (group_id = 0)
    singletons = entity_groups[entity_groups['group_id'] == 0]['entity_id'].tolist()
    entities_to_keep.extend(singletons)

    # Keep first entity from each duplicate group
    duplicate_groups = entity_groups[entity_groups['group_id'] > 0].groupby('group_id')
    for group_id, group_df in duplicate_groups:
        first_entity = group_df.iloc[0]['entity_id']
        entities_to_keep.append(first_entity)

    return original_df[original_df[id_column].isin(entities_to_keep)].copy()