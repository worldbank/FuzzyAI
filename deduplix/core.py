from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
import networkx as nx
from pathlib import Path
import json
import hashlib
from datetime import datetime
from abc import ABC, abstractmethod
from .exceptions import ConfigurationError, DataValidationError
from .utils import remove_duplicates_after_deduplication  # noqa: F401  (re-exported)

_UNSET = object()

@dataclass
class MatchResult:
    """Result from matching stage"""
    pairs: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result from validation stage"""
    validated_pairs: pd.DataFrame
    removed_pairs: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeduplicationResult:
    """Final deduplication result"""
    entity_groups: pd.DataFrame
    duplicate_pairs: pd.DataFrame
    statistics: Dict[str, Any]
    
    def get_group(self, entity_id) -> List:
        """Get all entities in the same group as entity_id"""
        group_data = self.entity_groups[self.entity_groups['entity_id'] == entity_id]
        if group_data.empty:
            return []
        group_id = group_data['group_id'].iloc[0]
        if group_id == 0:  # group_id 0 marks singletons, which share no group
            return [entity_id]
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
        
        singletons = entity_groups[entity_groups['group_id'] == 0]['entity_id'].tolist()
        entities_to_keep.extend(singletons)
        
        duplicate_groups = entity_groups[entity_groups['group_id'] > 0].groupby('group_id')
        
        if keep_strategy in ('first', 'last'):
            positions = pd.Series(range(len(original_df)))
            agg = 'min' if keep_strategy == 'first' else 'max'
            pos_by_id = positions.groupby(original_df[id_column].to_numpy()).agg(agg)

        for group_id, group_df in duplicate_groups:
            group_entities = group_df['entity_id'].tolist()

            if keep_strategy in ('first', 'last'):
                found = pos_by_id.reindex(group_entities).dropna()
                if len(found):
                    pick = found.idxmin() if keep_strategy == 'first' else found.idxmax()
                    entities_to_keep.append(pick)

            elif keep_strategy == 'highest_score':
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
                entities_to_keep.append(group_entities[0])
        
        cleaned_df = original_df[original_df[id_column].isin(entities_to_keep)].copy()
        return cleaned_df

    def get_entities_to_keep(self, keep_strategy: str = 'first') -> List:
        """Get list of entity IDs to keep (one from each duplicate group + singletons)"""
        entities_to_keep = []
        
        singletons = self.entity_groups[self.entity_groups['group_id'] == 0]['entity_id'].tolist()
        entities_to_keep.extend(singletons)
        
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
    cross_matches: pd.DataFrame
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
            
        id_col1 = self.df1_metadata['id_col']
        id_col2 = self.df2_metadata['id_col'] 
        
        merge_keys = self.cross_matches[['df1_id', 'df2_id', 'similarity_score']].copy()
        merge_keys = merge_keys.rename(columns={'df1_id': id_col1, 'df2_id': id_col2})
        
        result = merge_keys.merge(df1, on=id_col1, how=how, suffixes=('', suffix1))
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
        config,
        id_column1: Optional[str] = None,
        id_column2: Optional[str] = None,
        name_column2: Optional[str] = None
    ) -> pd.DataFrame:
        """Consolidate matches using ConsolidationConfig"""
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
        config,
        id_column1: Optional[str] = None,
        id_column2: Optional[str] = None,
        name_column2: Optional[str] = None,
        checkpointer=None,
        data_hash: Optional[str] = None,
        resume: bool = True
    ) -> pd.DataFrame:
        """Consolidate matches using LLM-based intelligent selection"""
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
        checkpoint_dir: str = ".deduplix_checkpoints",
        checkpoint_type=_UNSET,
        checkpoint_db_path: Optional[str] = None,
        checkpoint_compress: bool = False,
        hooks=None,
    ):
        self.matcher = matcher
        self.validator = validator
        self.hooks = list(hooks or [])
        self.checkpoint_enabled = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.checkpointer = None
        if checkpoint_type is _UNSET:
            # legacy behaviour: keep the old checkpoint layout so existing runs still resume
            self.checkpoint_type = "file"
            if checkpoint:
                self.checkpointer = Checkpointer(checkpoint_dir)
        else:
            self.checkpoint_type = checkpoint_type
            if checkpoint_type == "file":
                from .checkpointing import FileCheckpointer
                if checkpoint:
                    self.checkpointer = FileCheckpointer(checkpoint_dir, compress=checkpoint_compress)
            elif checkpoint_type == "database":
                from .checkpointing import DatabaseCheckpointer
                if checkpoint:
                    self.checkpointer = DatabaseCheckpointer(
                        checkpoint_db_path or "deduplix_checkpoints.db", compress=checkpoint_compress)
            else:
                raise ConfigurationError(f"Unsupported checkpoint type: {checkpoint_type}",
                                         config_key="checkpoint_type", config_value=checkpoint_type)
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute collision-resistant hash of dataframe for checkpoint identification"""
        cols = [c for c in ('id', 'name') if c in df.columns]
        row_hash = pd.util.hash_pandas_object(df[cols].astype(str), index=False).sum()
        hash_str = f"{df.shape[0]}_{df.shape[1]}_{row_hash}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:16]
    
    def _remove_symmetric_pairs(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Remove symmetric duplicate pairs (A,B) and (B,A)"""
        if pairs_df.empty:
            return pairs_df
            
        df = pairs_df.copy()
        df['canonical_id1'] = df[['id1', 'id2']].min(axis=1)
        df['canonical_id2'] = df[['id1', 'id2']].max(axis=1)
        df = df.drop_duplicates(subset=['canonical_id1', 'canonical_id2'])
        df = df.drop(columns=['canonical_id1', 'canonical_id2'])
        
        return df
    
    def _cluster_entities(self, validated_pairs: pd.DataFrame, all_entities: pd.DataFrame) -> DeduplicationResult:
        """Create groups using connected components"""
        
        G = nx.Graph()
        
        if not validated_pairs.empty:
            weights = (validated_pairs['similarity_score'] if 'similarity_score' in validated_pairs.columns
                       else pd.Series(1.0, index=validated_pairs.index))
            G.add_weighted_edges_from(
                zip(validated_pairs['id1'], validated_pairs['id2'], weights), weight='weight')
        
        components = list(nx.connected_components(G))
        
        entity_to_group = {}
        for group_id, component in enumerate(components, start=1):
            for entity_id in component:
                entity_to_group[entity_id] = group_id
        
        entity_groups = pd.DataFrame({
            'entity_id': all_entities['id'].to_numpy(),
            'entity_name': all_entities['name'].to_numpy(),
            'group_id': all_entities['id'].map(entity_to_group).fillna(0).astype('int64').to_numpy(),
        })
        
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
        """Run the complete deduplication pipeline"""
        
        if id_column not in df.columns or name_column not in df.columns:
            raise DataValidationError(f"Required columns {id_column} and {name_column} not found")
        
        working_df = df.rename(columns={id_column: 'id', name_column: 'name'})
        data_hash = self._compute_data_hash(working_df) if self.checkpoint_enabled else None
        
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
        
        match_result.pairs = self._remove_symmetric_pairs(match_result.pairs)
        print(f"  After removing symmetric pairs: {len(match_result.pairs)} pairs")
        
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
        
        if not hasattr(self.matcher, 'find_cross_matches'):
            raise ValueError("Matcher does not support cross-dataset matching")
        
        print(f"Cross-dataset matching: {len(df1)} entities in df1 vs {len(df2)} entities in df2")
        
        data_hash = None
        if self.checkpoint_enabled:
            df1_for_hash = df1.rename(columns={id_column1: 'id', name_column1: 'name'})
            df2_for_hash = df2.rename(columns={id_column2: 'id', name_column2: 'name'})
            
            hash1 = self._compute_data_hash(df1_for_hash)
            hash2 = self._compute_data_hash(df2_for_hash)
            data_hash = f"{hash1}_{hash2}"
        
        print(f"Stage 1: Finding cross-dataset matches...")
        
        if self.checkpoint_enabled and resume:
            match_result_df = self.checkpointer.load('cross_matching', data_hash)
            if match_result_df is not None:
                print(f"  Loaded {len(match_result_df)} matches from checkpoint")
                match_result = MatchResult(pairs=match_result_df)
            else:
                match_result = self.matcher.find_cross_matches(
                    df1, df2,
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
                df1, df2,
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
        
        if self.validator:
            print(f"Stage 2: Validating cross-dataset matches...")
            
            combined_df = pd.concat([
                df1.rename(columns={id_column1: 'id', name_column1: 'name'}),
                df2.rename(columns={id_column2: 'id', name_column2: 'name'})
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
        
        result = self._create_cross_dataset_result(df1, df2, pairs_for_result, 
                                                id_column1, name_column1, 
                                                id_column2, name_column2)
        
        print(f"\nCross-dataset matching complete:")
        print(f"  DF1 entities: {len(df1)}")
        print(f"  DF2 entities: {len(df2)}")
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
        
        if validated_pairs.empty:
            matches_df = pd.DataFrame()
        else:
            vp = validated_pairs
            matches_df = pd.DataFrame({
                'df1_id': vp['id1'].to_numpy(),
                'df1_name': vp['name1'].to_numpy(),
                'df2_id': vp['id2'].to_numpy(),
                'df2_name': vp['name2'].to_numpy(),
                'similarity_score': vp['similarity_score'].to_numpy(),
                'validation_reason': (vp['validation_reason'].to_numpy() if 'validation_reason' in vp.columns
                                      else ''),
            })
            for col in ('matched_column1', 'matched_column2'):
                if col in vp.columns:
                    matches_df[col] = vp[col].to_numpy()

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