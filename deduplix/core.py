

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
import networkx as nx
from pathlib import Path
import json
import hashlib
from datetime import datetime
from abc import ABC, abstractmethod


@dataclass
class MatchResult:
    """Result from matching stage"""
    pairs: pd.DataFrame  # columns: [id1, id2, name1, name2, similarity_score]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result from validation stage"""
    validated_pairs: pd.DataFrame  # same structure as MatchResult.pairs + validation_reason
    removed_pairs: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeduplicationResult:
    """Final deduplication result"""
    entity_groups: pd.DataFrame  # columns: [entity_id, entity_name, group_id]
    duplicate_pairs: pd.DataFrame  # validated pairs used for grouping
    statistics: Dict[str, Any]
    
    def get_group(self, entity_id) -> List:
        """Get all entities in the same group as entity_id"""
        group_data = self.entity_groups[self.entity_groups['entity_id'] == entity_id]
        if group_data.empty:
            return []
        group_id = group_data['group_id'].iloc[0]
        return self.entity_groups[
            self.entity_groups['group_id'] == group_id
        ]['entity_id'].tolist()
    
    def save(self, path: str):
        """Save results to directory"""
        output_dir = Path(path)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        self.entity_groups.to_csv(output_dir / 'entity_groups.csv', index=False)
        self.duplicate_pairs.to_csv(output_dir / 'duplicate_pairs.csv', index=False)
        
        with open(output_dir / 'statistics.json', 'w') as f:
            json.dump(self.statistics, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str):
        """Load results from directory"""
        input_dir = Path(path)
        
        entity_groups = pd.read_csv(input_dir / 'entity_groups.csv')
        duplicate_pairs = pd.read_csv(input_dir / 'duplicate_pairs.csv')
        
        with open(input_dir / 'statistics.json', 'r') as f:
            statistics = json.load(f)
        
        return cls(
            entity_groups=entity_groups,
            duplicate_pairs=duplicate_pairs,
            statistics=statistics
        )


class Matcher(ABC):
    """Abstract base class for matchers"""
    
    @abstractmethod
    def find_matches(self, df: pd.DataFrame, **kwargs) -> MatchResult:
        pass


class Validator(ABC):
    """Abstract base class for validators"""
    
    @abstractmethod
    def validate(self, match_result: MatchResult, **kwargs) -> ValidationResult:
        pass


class Checkpointer:
    """Handles saving and loading checkpoints"""
    
    def __init__(self, checkpoint_dir: str = ".deduplix_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def get_checkpoint_path(self, stage: str, data_hash: str) -> Path:
        """Generate checkpoint file path"""
        return self.checkpoint_dir / f"{stage}_{data_hash}.parquet"
    
    def save(self, data: pd.DataFrame, stage: str, data_hash: str):
        """Save checkpoint"""
        path = self.get_checkpoint_path(stage, data_hash)
        data.to_parquet(path)
        
        # Save metadata
        meta_path = path.with_suffix('.meta.json')
        metadata = {
            'stage': stage,
            'data_hash': data_hash,
            'timestamp': datetime.now().isoformat(),
            'rows': len(data)
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, stage: str, data_hash: str) -> Optional[pd.DataFrame]:
        """Load checkpoint if exists"""
        path = self.get_checkpoint_path(stage, data_hash)
        if path.exists():
            return pd.read_parquet(path)
        return None
    
    def clear(self):
        """Clear all checkpoints"""
        for file in self.checkpoint_dir.glob("*"):
            file.unlink()


class DeduplicationPipeline:
    """Main deduplication pipeline"""
    
    def __init__(
        self,
        matcher: Matcher,
        validator: Optional[Validator] = None,
        checkpoint: bool = True,
        checkpoint_dir: str = ".deduplix_checkpoints"
    ):
        self.matcher = matcher
        self.validator = validator
        self.checkpoint_enabled = checkpoint
        self.checkpointer = Checkpointer(checkpoint_dir) if checkpoint else None
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe for checkpoint identification"""
        # Use shape and sample of data for hash
        hash_str = f"{df.shape}_{df.iloc[:min(100, len(df))].to_json()}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:12]
    
    def _remove_symmetric_pairs(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Remove symmetric duplicate pairs (A,B) and (B,A)"""
        if pairs_df.empty:
            return pairs_df
            
        df = pairs_df.copy()
        
        # Create canonical form where smaller ID is always first
        df['canonical_id1'] = df[['id1', 'id2']].min(axis=1)
        df['canonical_id2'] = df[['id1', 'id2']].max(axis=1)
        
        # Remove duplicates based on canonical form
        df = df.drop_duplicates(subset=['canonical_id1', 'canonical_id2'])
        
        # Clean up temporary columns
        df = df.drop(columns=['canonical_id1', 'canonical_id2'])
        
        return df
    
    def _cluster_entities(self, validated_pairs: pd.DataFrame, all_entities: pd.DataFrame) -> DeduplicationResult:
        """Create groups using connected components"""
        
        # Build graph from validated pairs
        G = nx.Graph()
        
        if not validated_pairs.empty:
            for _, row in validated_pairs.iterrows():
                G.add_edge(row['id1'], row['id2'], weight=row.get('similarity_score', 1.0))
        
        # Find connected components
        components = list(nx.connected_components(G))
        
        # Create entity to group mapping
        entity_to_group = {}
        for group_id, component in enumerate(components, start=1):
            for entity_id in component:
                entity_to_group[entity_id] = group_id
        
        # Build result dataframe including entities with no duplicates
        result_rows = []
        
        # Add all entities, assigning group 0 to singletons
        for _, row in all_entities.iterrows():
            entity_id = row['id']
            entity_name = row['name']
            group_id = entity_to_group.get(entity_id, 0)  # 0 for no duplicates
            
            result_rows.append({
                'entity_id': entity_id,
                'entity_name': entity_name,
                'group_id': group_id
            })
        
        entity_groups = pd.DataFrame(result_rows)
        
        # Calculate statistics
        statistics = {
            'total_entities': len(all_entities),
            'duplicate_pairs': len(validated_pairs),
            'duplicate_groups': len(components),
            'entities_with_duplicates': len(entity_to_group),
            'singleton_entities': len(all_entities) - len(entity_to_group),
            'average_group_size': len(entity_to_group) / len(components) if components else 0,
            'largest_group_size': max(len(c) for c in components) if components else 0
        }
        
        return DeduplicationResult(
            entity_groups=entity_groups,
            duplicate_pairs=validated_pairs,
            statistics=statistics
        )
    
    def run(
        self,
        df: pd.DataFrame,
        id_column: str = 'id',
        name_column: str = 'name',
        additional_columns: Optional[List[str]] = None,
        resume: bool = True
    ) -> DeduplicationResult:
        """
        Run the complete deduplication pipeline
        
        Args:
            df: Input dataframe with entities
            id_column: Column name for entity ID
            name_column: Column name for entity name/description
            additional_columns: Additional columns to consider in matching
            resume: Whether to resume from checkpoint if available
        """
        
        # Ensure required columns exist
        if id_column not in df.columns or name_column not in df.columns:
            raise ValueError(f"Required columns {id_column} and {name_column} not found")
        
        # Standardize column names
        working_df = df.rename(columns={id_column: 'id', name_column: 'name'})
        
        data_hash = self._compute_data_hash(working_df) if self.checkpoint_enabled else None
        
        # Stage 1: Matching
        print(f"Stage 1: Finding potential matches...")
        
        if self.checkpoint_enabled and resume:
            match_result_df = self.checkpointer.load('matching', data_hash)
            if match_result_df is not None:
                print(f"  Loaded {len(match_result_df)} matches from checkpoint")
                match_result = MatchResult(pairs=match_result_df)
            else:
                match_result = self.matcher.find_matches(working_df, additional_columns=additional_columns)
                if not match_result.pairs.empty:
                    self.checkpointer.save(match_result.pairs, 'matching', data_hash)
        else:
            match_result = self.matcher.find_matches(working_df, additional_columns=additional_columns)
            if self.checkpoint_enabled and not match_result.pairs.empty:
                self.checkpointer.save(match_result.pairs, 'matching', data_hash)
        
        print(f"  Found {len(match_result.pairs)} potential duplicate pairs")
        
        # Remove symmetric pairs
        match_result.pairs = self._remove_symmetric_pairs(match_result.pairs)
        print(f"  After removing symmetric pairs: {len(match_result.pairs)} pairs")
        
        # Stage 2: Validation (optional)
        if self.validator:
            print(f"Stage 2: Validating matches...")
            
            if self.checkpoint_enabled and resume:
                validated_df = self.checkpointer.load('validation', data_hash)
                if validated_df is not None:
                    print(f"  Loaded {len(validated_df)} validated pairs from checkpoint")
                    validation_result = ValidationResult(
                        validated_pairs=validated_df,
                        removed_pairs=pd.DataFrame()
                    )
                else:
                    validation_result = self.validator.validate(match_result)
                    if not validation_result.validated_pairs.empty:
                        self.checkpointer.save(validation_result.validated_pairs, 'validation', data_hash)
            else:
                validation_result = self.validator.validate(match_result)
                if self.checkpoint_enabled and not validation_result.validated_pairs.empty:
                    self.checkpointer.save(validation_result.validated_pairs, 'validation', data_hash)
            
            print(f"  Validated: {len(validation_result.validated_pairs)} pairs kept, "
                  f"{len(validation_result.removed_pairs)} removed")
            
            pairs_for_clustering = validation_result.validated_pairs
        else:
            pairs_for_clustering = match_result.pairs
        
        # Stage 3: Clustering
        print(f"Stage 3: Clustering entities into groups...")
        result = self._cluster_entities(pairs_for_clustering, working_df)
        
        print(f"\nDeduplication Complete:")
        print(f"  Total entities: {result.statistics['total_entities']}")
        print(f"  Duplicate groups found: {result.statistics['duplicate_groups']}")
        print(f"  Entities with duplicates: {result.statistics['entities_with_duplicates']}")
        print(f"  Singleton entities: {result.statistics['singleton_entities']}")
        
        return result