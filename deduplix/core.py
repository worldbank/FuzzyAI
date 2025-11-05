from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
import networkx as nx
from pathlib import Path
import json
import hashlib
from datetime import datetime
from abc import ABC, abstractmethod
import warnings
from .utils import remove_duplicates_after_deduplication
from .validation import validate_deduplication_input, validate_cross_dataset_input
from .exceptions import (
    DeduplixError, DataValidationError, MatchingError, ValidationError,
    CheckpointError, ConfigurationError, handle_and_reraise
)
from .checkpointing import BaseCheckpointer, FileCheckpointer, DatabaseCheckpointer

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
    
    def remove_duplicates(
        self, 
        original_df: pd.DataFrame,
        id_column: str = 'id',
        keep_strategy: str = 'first'
    ) -> pd.DataFrame:
        """Remove duplicate rows from original dataframe"""
        
        entity_groups = self.entity_groups.copy()
        entities_to_keep = []
        
        # Keep singletons (group_id = 0)
        singletons = entity_groups[entity_groups['group_id'] == 0]['entity_id'].tolist()
        entities_to_keep.extend(singletons)
        
        # Process duplicate groups
        duplicate_groups = entity_groups[entity_groups['group_id'] > 0].groupby('group_id')
        
        for group_id, group_df in duplicate_groups:
            group_entities = group_df['entity_id'].tolist()
            
            if keep_strategy == 'first':
                # Keep entity that appears first in original dataframe
                original_indices = []
                for entity_id in group_entities:
                    idx = original_df[original_df[id_column] == entity_id].index
                    if len(idx) > 0:
                        original_indices.append((entity_id, idx[0]))
                
                if original_indices:
                    original_indices.sort(key=lambda x: x[1])
                    entities_to_keep.append(original_indices[0][0])
                    
            elif keep_strategy == 'last':
                # Keep entity that appears last in original dataframe
                original_indices = []
                for entity_id in group_entities:
                    idx = original_df[original_df[id_column] == entity_id].index
                    if len(idx) > 0:
                        original_indices.append((entity_id, idx[-1]))
                
                if original_indices:
                    original_indices.sort(key=lambda x: x[1])
                    entities_to_keep.append(original_indices[-1][0])
                    
            elif keep_strategy == 'highest_score':
                # Keep entity with highest average similarity score
                if not self.duplicate_pairs.empty:
                    entity_scores = {}
                    for entity_id in group_entities:
                        pairs_as_id1 = self.duplicate_pairs[self.duplicate_pairs['id1'] == entity_id]
                        pairs_as_id2 = self.duplicate_pairs[self.duplicate_pairs['id2'] == entity_id]
                        
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
                # Default to first entity in group
                entities_to_keep.append(group_entities[0])
        
        # Filter original dataframe
        cleaned_df = original_df[original_df[id_column].isin(entities_to_keep)].copy()
        return cleaned_df

    def get_entities_to_keep(self, keep_strategy: str = 'first') -> List:
        """Get list of entity IDs to keep (one from each duplicate group + singletons)"""
        entities_to_keep = []
        
        # Keep singletons (group_id = 0)
        singletons = self.entity_groups[self.entity_groups['group_id'] == 0]['entity_id'].tolist()
        entities_to_keep.extend(singletons)
        
        # One entity from each duplicate group
        duplicate_groups = self.entity_groups[self.entity_groups['group_id'] > 0].groupby('group_id')
        
        for group_id, group_df in duplicate_groups:
            group_entities = group_df['entity_id'].tolist()
            
            if keep_strategy == 'highest_score' and not self.duplicate_pairs.empty:
                entity_scores = {}
                for entity_id in group_entities:
                    pairs_as_id1 = self.duplicate_pairs[self.duplicate_pairs['id1'] == entity_id]
                    pairs_as_id2 = self.duplicate_pairs[self.duplicate_pairs['id2'] == entity_id]
                    
                    scores = []
                    scores.extend(pairs_as_id1['similarity_score'].tolist())
                    scores.extend(pairs_as_id2['similarity_score'].tolist())
                    
                    entity_scores[entity_id] = sum(scores) / len(scores) if scores else 0
                
                best_entity = max(entity_scores.keys(), key=lambda x: entity_scores[x])
                entities_to_keep.append(best_entity)
            else:
                entities_to_keep.append(group_entities[0])
        
        return entities_to_keep

    def get_removal_summary(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of removal process"""
        return {
            'original_count': len(original_df),
            'cleaned_count': len(cleaned_df),
            'duplicates_removed': len(original_df) - len(cleaned_df),
            'duplicate_groups_found': self.statistics['duplicate_groups'],
            'entities_with_duplicates': self.statistics['entities_with_duplicates'],
            'removal_rate': (len(original_df) - len(cleaned_df)) / len(original_df) * 100 if len(original_df) > 0 else 0
        }


@dataclass
class CrossDatasetResult:
    """Result from cross-dataset matching"""
    cross_matches: pd.DataFrame  # columns: [df1_id, df1_name, df2_id, df2_name, similarity_score]
    df1_metadata: Dict[str, Any]
    df2_metadata: Dict[str, Any] 
    statistics: Dict[str, Any]
    
    def get_df1_matches(self, df1_id) -> List[Dict]:
        """Get all df2 entities that match a df1 entity"""
        matches = self.cross_matches[self.cross_matches['df1_id'] == df1_id]
        return matches.to_dict('records')
    
    def get_df2_matches(self, df2_id) -> List[Dict]:
        """Get all df1 entities that match a df2 entity"""  
        matches = self.cross_matches[self.cross_matches['df2_id'] == df2_id]
        return matches.to_dict('records')
    
    def merge_datasets(
                self, 
                df1: pd.DataFrame, 
                df2: pd.DataFrame,
                how: str = 'inner',
                suffix1: str = '_df1',
                suffix2: str = '_df2'
            ) -> pd.DataFrame:
        """Merge datasets based on found matches"""
        
        if self.cross_matches.empty:
            return pd.DataFrame()
            
        # Prepare merge keys
        id_col1 = self.df1_metadata['id_col']
        id_col2 = self.df2_metadata['id_col'] 
        
        # Create merge dataframe
        merge_keys = self.cross_matches[['df1_id', 'df2_id', 'similarity_score']].copy()
        merge_keys = merge_keys.rename(columns={'df1_id': id_col1, 'df2_id': id_col2})
        
        # Merge df1
        result = merge_keys.merge(df1, on=id_col1, how=how, suffixes=('', suffix1))
        
        # Merge df2
        result = result.merge(df2, on=id_col2, how=how, suffixes=(suffix1, suffix2))
        
        return result
    
    def save(self, path: str):
        """Save cross-dataset results"""
        output_dir = Path(path)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        self.cross_matches.to_csv(output_dir / 'cross_matches.csv', index=False)
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump({
                'df1_metadata': self.df1_metadata,
                'df2_metadata': self.df2_metadata, 
                'statistics': self.statistics
            }, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str):
        """Load cross-dataset results"""
        input_dir = Path(path)
        
        cross_matches = pd.read_csv(input_dir / 'cross_matches.csv')
        
        with open(input_dir / 'metadata.json', 'r') as f:
            data = json.load(f)
        
        return cls(
            cross_matches=cross_matches,
            df1_metadata=data['df1_metadata'],
            df2_metadata=data['df2_metadata'],
            statistics=data['statistics']
        )
    def consolidate_with_config(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        config,  # ConsolidationConfig
        id_column1: Optional[str] = None,
        id_column2: Optional[str] = None,
        name_column2: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Consolidate matches using ConsolidationConfig.
        
        For each entity in df1, this selects ONE main match from df2 based on
        priority rules, and lists all other match IDs in 'other_candidates' column.
        
        Parameters
        ----------
        df1 : pd.DataFrame
            First dataset
        df2 : pd.DataFrame
            Second dataset
        config : ConsolidationConfig
            Consolidation configuration (from consolidation.py)
        id_column1 : str, optional
            ID column in df1 (uses metadata if None)
        id_column2 : str, optional
            ID column in df2 (uses metadata if None)
        name_column2 : str, optional
            Name column in df2 (uses metadata if None)
            
        Returns
        -------
        pd.DataFrame
            Consolidated dataframe with columns:
            - All original df1 columns
            - main_{id_column2}: Selected match ID
            - main_{name_column2}: Selected match name
            - main_similarity_score: Match quality
            - main_{priority_column}: Priority source (if configured)
            - other_candidates: Comma-separated other match IDs
            - total_matches: Number of matches found
            - all_match_ids: All match IDs (if keep_all_matches=True)
            
     
        """
        from .consolidation import ConsolidationEngine
        
        id_col1 = id_column1 or self.df1_metadata['id_col']
        id_col2 = id_column2 or self.df2_metadata['id_col']
        name_col2 = name_column2 or self.df2_metadata['name_col']
        
        engine = ConsolidationEngine(config)
        return engine.consolidate(
            self.cross_matches,
            df1,
            df2,
            id_col1,
            id_col2,
            name_col2
        )
    def consolidate_with_llm(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        config,  # LLMConsolidationConfig
        id_column1: Optional[str] = None,
        id_column2: Optional[str] = None,
        name_column2: Optional[str] = None,
        checkpointer=None,
        data_hash: Optional[str] = None,
        resume: bool = True
    ) -> pd.DataFrame:
        """
        Consolidate matches using LLM-based intelligent selection.
        
        Uses an LLM to select the best match for each entity when multiple
        candidates exist. The LLM considers metadata, similarity scores,
        and custom instructions to make informed decisions.
        
        Parameters
        ----------
        df1 : pd.DataFrame
            First dataset
        df2 : pd.DataFrame
            Second dataset
        config : LLMConsolidationConfig
            LLM consolidation configuration (from llm_consolidation.py)
        id_column1 : str, optional
            ID column in df1 (uses metadata if None)
        id_column2 : str, optional
            ID column in df2 (uses metadata if None)
        name_column2 : str, optional
            Name column in df2 (uses metadata if None)
        checkpointer : Checkpointer, optional
            Checkpointer instance for saving progress
        data_hash : str, optional
            Hash for checkpoint files
        resume : bool
            Whether to resume from checkpoint (default: True)
            
        Returns
        -------
        pd.DataFrame
            Consolidated dataframe with columns:
            - All original df1 columns
            - main_{id_column2}: LLM-selected match ID
            - main_{name_column2}: LLM-selected match name
            - main_similarity_score: Match quality
            - llm_reasoning: LLM's explanation for selection
            - other_candidates: Comma-separated other match IDs
            - total_matches: Number of matches found
            
       
        """
        from .llm_consolidation import LLMConsolidationEngine
        
        id_col1 = id_column1 or self.df1_metadata['id_col']
        id_col2 = id_column2 or self.df2_metadata['id_col']
        name_col2 = name_column2 or self.df2_metadata['name_col']
        
        engine = LLMConsolidationEngine(config)
        return engine.consolidate(
            self.cross_matches,
            df1,
            df2,
            id_col1,
            id_col2,
            name_col2,
            checkpointer=checkpointer,
            data_hash=data_hash,
            resume=resume
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


# Legacy Checkpointer class removed - now using modular checkpointing system
# See deduplix.checkpointing module for FileCheckpointer and DatabaseCheckpointer


class DeduplicationPipeline:
    """Main deduplication pipeline"""
    
    def __init__(
        self,
        matcher: Matcher,
        validator: Optional[Validator] = None,
        checkpoint: bool = True,
        checkpoint_type: str = "file",  # "file" or "database"
        checkpoint_dir: str = ".deduplix_checkpoints",
        checkpoint_db_path: str = "deduplix_checkpoints.db",
        checkpoint_compress: bool = True
    ):
        """
        Initialize deduplication pipeline

        Parameters
        ----------
        matcher : Matcher
            Entity matching algorithm
        validator : Optional[Validator]
            Match validation algorithm
        checkpoint : bool
            Whether to enable checkpointing
        checkpoint_type : str
            Type of checkpointing: "file" or "database"
        checkpoint_dir : str
            Directory for file-based checkpoints
        checkpoint_db_path : str
            Database path for database checkpoints
        checkpoint_compress : bool
            Whether to compress checkpoint data
        """
        self.matcher = matcher
        self.validator = validator
        self.checkpoint_enabled = checkpoint
        self.checkpoint_type = checkpoint_type

        # Initialize appropriate checkpointer
        if checkpoint:
            if checkpoint_type == "database":
                self.checkpointer = DatabaseCheckpointer(
                    db_path=checkpoint_db_path,
                    compress=checkpoint_compress
                )
            elif checkpoint_type == "file":
                self.checkpointer = FileCheckpointer(
                    checkpoint_dir=checkpoint_dir,
                    compress=checkpoint_compress
                )
            else:
                raise ConfigurationError(
                    f"Unsupported checkpoint type: {checkpoint_type}",
                    config_key="checkpoint_type",
                    config_value=checkpoint_type,
                    suggestions=[
                        "Use 'file' for file-based checkpointing",
                        "Use 'database' for SQLite database checkpointing"
                    ]
                )
        else:
            self.checkpointer = None
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """
        Compute collision-resistant hash of dataframe for checkpoint identification.
        
        Uses shape + hash of all entity IDs to ensure different datasets get
        different hashes, even if they have the same number of rows.
        """
        # Hash all IDs - this makes the hash unique to the specific entities
        id_hash = pd.util.hash_pandas_object(df['id'], index=False).sum()
        
    
        hash_str = f"{df.shape[0]}_{df.shape[1]}_{id_hash}"
        
        return hashlib.md5(hash_str.encode()).hexdigest()[:16]
    
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
        
        # Comprehensive input validation
        print("Stage 0: Validating input data...")
        try:
            validated_df, validation_metadata = validate_deduplication_input(
                df=df,
                id_column=id_column,
                name_column=name_column,
                additional_columns=additional_columns
            )
            print(f"  Validated {validation_metadata['final_row_count']} entities "
                  f"(data quality score: {validation_metadata['data_quality_score']:.2f})")

            if validation_metadata['validation_warnings']:
                print(f"  Warnings: {'; '.join(validation_metadata['validation_warnings'])}")

        except Exception as e:
            handle_and_reraise(
                func_name="DeduplicationPipeline.run.validation",
                original_exception=e,
                context={
                    'input_shape': df.shape,
                    'id_column': id_column,
                    'name_column': name_column
                },
                suggestions=[
                    "Check input DataFrame format and column names",
                    "Ensure ID and name columns exist and contain valid data",
                    "Review data quality and remove invalid entries"
                ]
            )

        # Standardize column names for processing
        working_df = validated_df.rename(columns={id_column: 'id', name_column: 'name'})
        
        data_hash = self._compute_data_hash(working_df) if self.checkpoint_enabled else None
        
        # Stage 1: Matching
        print(f"Stage 1: Finding potential matches...")

        try:
            if self.checkpoint_enabled and resume:
                try:
                    match_result_df = self.checkpointer.load('matching', data_hash)
                    if match_result_df is not None:
                        print(f"  Loaded {len(match_result_df)} matches from checkpoint")
                        match_result = MatchResult(pairs=match_result_df)
                    else:
                        match_result = self.matcher.find_matches(working_df, additional_columns=additional_columns)
                        if not match_result.pairs.empty:
                            self.checkpointer.save(match_result.pairs, 'matching', data_hash)
                except Exception as e:
                    raise CheckpointError(
                        f"Failed to load matching checkpoint: {e}",
                        checkpoint_stage='matching',
                        context={'data_hash': data_hash}
                    ) from e
            else:
                match_result = self.matcher.find_matches(working_df, additional_columns=additional_columns)
                if self.checkpoint_enabled and not match_result.pairs.empty:
                    try:
                        self.checkpointer.save(match_result.pairs, 'matching', data_hash)
                    except Exception as e:
                        warnings.warn(f"Failed to save matching checkpoint: {e}", UserWarning)

        except MatchingError:
            raise  # Re-raise matching errors as-is
        except Exception as e:
            handle_and_reraise(
                func_name="DeduplicationPipeline.run.matching",
                original_exception=e,
                context={
                    'entity_count': len(working_df),
                    'matcher_type': type(self.matcher).__name__
                },
                suggestions=[
                    "Check matching configuration and parameters",
                    "Verify entity data format and completeness",
                    "Consider reducing dataset size for troubleshooting"
                ]
            )
        
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
                    # With checkpoint parameters
                    validation_result = self.validator.validate(
                        match_result,
                        original_df=working_df,
                        checkpointer=self.checkpointer,
                        data_hash=data_hash,
                        resume=resume
                    )
                    if not validation_result.validated_pairs.empty:
                        self.checkpointer.save(validation_result.validated_pairs, 'validation', data_hash)
            else:
                # : with checkpoint parameters
                validation_result = self.validator.validate(
                    match_result,
                    original_df=working_df,
                    checkpointer=self.checkpointer,
                    data_hash=data_hash,
                    resume=resume
                )
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


    def run_cross_dataset(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        id_column1: str = 'id',
        name_column1: str = 'name',
        id_column2: str = 'id', 
        name_column2: str = 'name',
        name_columns1: Optional[List[str]] = None,  
        name_columns2: Optional[List[str]] = None,  
        additional_columns1: Optional[List[str]] = None,
        additional_columns2: Optional[List[str]] = None,
        resume: bool = True
    ) -> CrossDatasetResult:
        """
        Run cross-dataset deduplication between two dataframes with checkpoint support.
        
        Parameters
        ----------
        df1, df2 : pd.DataFrame
            Datasets to match
        id_column1, id_column2 : str
            ID column names
        name_column1, name_column2 : str
            Primary name columns (used if name_columns1/2 not specified)
        name_columns1 : List[str], optional
            Multiple name columns to try for df1 (e.g., ['legal_name', 'short_name'])
            Will try all combinations and keep best match
        name_columns2 : List[str], optional
            Multiple name columns to try for df2
        additional_columns1, additional_columns2 : List[str], optional
            Additional columns to consider
        resume : bool
            Whether to resume from checkpoint
            
        Returns
        -------
        CrossDatasetResult
            Results including which columns produced each match
            
        Examples
        --------
        # Match using multiple name columns
        >>> result = pipeline.run_cross_dataset(
        ...     df1, df2,
        ...     id_column1='company_id',
        ...     name_columns1=['legal_name', 'short_name', 'dba'],
        ...     id_column2='entity_id',
        ...     name_columns2=['name', 'trading_name']
        ... )
        """
        
        # Check if matcher supports cross-dataset matching
        if not hasattr(self.matcher, 'find_cross_matches'):
            raise ValueError("Matcher does not support cross-dataset matching")
        
        # Comprehensive input validation for cross-dataset matching
        print("Stage 0: Validating input datasets...")
        try:
            df1_validated, df2_validated, cross_validation_metadata = validate_cross_dataset_input(
                df1=df1, df2=df2,
                id_column1=id_column1, name_column1=name_column1,
                id_column2=id_column2, name_column2=name_column2,
                name_columns1=name_columns1, name_columns2=name_columns2
            )
            print(f"  DF1: {cross_validation_metadata['df1_final_rows']} entities "
                  f"(quality: {cross_validation_metadata['df1_quality_score']:.2f})")
            print(f"  DF2: {cross_validation_metadata['df2_final_rows']} entities "
                  f"(quality: {cross_validation_metadata['df2_quality_score']:.2f})")

            if cross_validation_metadata['validation_warnings']:
                print(f"  Warnings: {'; '.join(cross_validation_metadata['validation_warnings'])}")

        except Exception as e:
            raise ValueError(f"Cross-dataset validation failed: {e}")

        print(f"Cross-dataset matching: {len(df1_validated)} entities in df1 vs {len(df2_validated)} entities in df2")

        # Compute combined data hash for checkpointing
        data_hash = None
        if self.checkpoint_enabled:
            # Standardize column names for hashing
            df1_for_hash = df1_validated.rename(columns={id_column1: 'id', name_column1: 'name'})
            df2_for_hash = df2_validated.rename(columns={id_column2: 'id', name_column2: 'name'})
            
            hash1 = self._compute_data_hash(df1_for_hash)
            hash2 = self._compute_data_hash(df2_for_hash)
            data_hash = f"{hash1}_{hash2}"
        
        # Stage 1: Cross-dataset matching
        print(f"Stage 1: Finding cross-dataset matches...")
        
        if self.checkpoint_enabled and resume:
            match_result_df = self.checkpointer.load('cross_matching', data_hash)
            if match_result_df is not None:
                print(f"  Loaded {len(match_result_df)} matches from checkpoint")
                match_result = MatchResult(pairs=match_result_df)
            else:
                match_result = self.matcher.find_cross_matches(
                    df1_validated, df2_validated,
                    id_column1=id_column1,
                    name_column1=name_column1,
                    id_column2=id_column2, 
                    name_column2=name_column2,
                    name_columns1=name_columns1,
                    name_columns2=name_columns2,
                    additional_columns1=additional_columns1,
                    additional_columns2=additional_columns2
                )
                if not match_result.pairs.empty:
                    self.checkpointer.save(match_result.pairs, 'cross_matching', data_hash)
        else:
            match_result = self.matcher.find_cross_matches(
                df1_validated, df2_validated,
                id_column1=id_column1,
                name_column1=name_column1,
                id_column2=id_column2, 
                name_column2=name_column2,
                name_columns1=name_columns1,
                name_columns2=name_columns2,
                additional_columns1=additional_columns1,
                additional_columns2=additional_columns2
            )
            if self.checkpoint_enabled and not match_result.pairs.empty:
                self.checkpointer.save(match_result.pairs, 'cross_matching', data_hash)
        
        print(f"  Found {len(match_result.pairs)} cross-dataset matches")
        
        # Stage 2: Validation (optional)
        if self.validator:
            print(f"Stage 2: Validating cross-dataset matches...")
            
            # For cross-dataset validation, combine both dataframes for context
            combined_df = pd.concat([
                df1_validated.rename(columns={id_column1: 'id', name_column1: 'name'}),
                df2_validated.rename(columns={id_column2: 'id', name_column2: 'name'})
            ], ignore_index=True)
            
            if self.checkpoint_enabled and resume:
                validated_df = self.checkpointer.load('cross_validation', data_hash)
                if validated_df is not None:
                    print(f"  Loaded {len(validated_df)} validated pairs from checkpoint")
                    validation_result = ValidationResult(
                        validated_pairs=validated_df,
                        removed_pairs=pd.DataFrame()
                    )
                else:
                    # With checkpoint parameters
                    validation_result = self.validator.validate(
                        match_result,
                        original_df=combined_df,
                        checkpointer=self.checkpointer,
                        data_hash=data_hash,
                        resume=resume
                    )
                    if not validation_result.validated_pairs.empty:
                        self.checkpointer.save(validation_result.validated_pairs, 'cross_validation', data_hash)
            else:
                # With  Added checkpoint parameters
                validation_result = self.validator.validate(
                    match_result,
                    original_df=combined_df,
                    checkpointer=self.checkpointer,
                    data_hash=data_hash,
                    resume=resume
                )
                if self.checkpoint_enabled and not validation_result.validated_pairs.empty:
                    self.checkpointer.save(validation_result.validated_pairs, 'cross_validation', data_hash)
            
            print(f"  Validated: {len(validation_result.validated_pairs)} pairs kept, "
                f"{len(validation_result.removed_pairs)} removed")
            
            pairs_for_result = validation_result.validated_pairs
        else:
            pairs_for_result = match_result.pairs
        
        # Create cross-dataset result
        result = self._create_cross_dataset_result(df1_validated, df2_validated, pairs_for_result,
                                                id_column1, name_column1,
                                                id_column2, name_column2)

        print(f"\nCross-dataset matching complete:")
        print(f"  DF1 entities: {len(df1_validated)}")
        print(f"  DF2 entities: {len(df2_validated)}")
        print(f"  Cross-matches found: {len(pairs_for_result)}")
        
        return result   
    def _create_cross_dataset_result(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        validated_pairs: pd.DataFrame,
        id_column1: str, 
        name_column1: str,
        id_column2: str, 
        name_column2: str
    ) -> CrossDatasetResult:
        """Create cross-dataset matching result"""
        
        # Create entity mapping
        matches = []
        for _, row in validated_pairs.iterrows():
            match_dict = {
                'df1_id': row['id1'],
                'df1_name': row['name1'],
                'df2_id': row['id2'], 
                'df2_name': row['name2'],
                'similarity_score': row['similarity_score'],
                'validation_reason': row.get('validation_reason', '')
            }
            # Add matched column info if available
            if 'matched_column1' in row:
                match_dict['matched_column1'] = row['matched_column1']
            if 'matched_column2' in row:
                match_dict['matched_column2'] = row['matched_column2']
            matches.append(match_dict)
        
        matches_df = pd.DataFrame(matches) if matches else pd.DataFrame()
        
        # Statistics
        df1_with_matches = set(validated_pairs['id1'].tolist()) if not validated_pairs.empty else set()
        df2_with_matches = set(validated_pairs['id2'].tolist()) if not validated_pairs.empty else set()
        
        statistics = {
            'df1_total': len(df1),
            'df2_total': len(df2),
            'cross_matches': len(validated_pairs),
            'df1_matched_entities': len(df1_with_matches),
            'df2_matched_entities': len(df2_with_matches),
            'df1_unmatched': len(df1) - len(df1_with_matches),
            'df2_unmatched': len(df2) - len(df2_with_matches),
            'avg_similarity': validated_pairs['similarity_score'].mean() if not validated_pairs.empty else 0
        }
        
        return CrossDatasetResult(
            cross_matches=matches_df,
            df1_metadata={'total': len(df1), 'id_col': id_column1, 'name_col': name_column1},
            df2_metadata={'total': len(df2), 'id_col': id_column2, 'name_col': name_column2},
            statistics=statistics
        )