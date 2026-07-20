
"""
Rule-based consolidation for cross-dataset matching results.

This module provides flexible configuration for selecting the "main" match
when an entity has multiple potential matches from cross-dataset matching.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
import pandas as pd


@dataclass
class ConsolidationConfig:
    """
    Configuration for priority-based consolidation of cross-dataset matches.
    
    This allows you to define rules for selecting the "main" match when an entity
    has multiple potential matches.
    
    Parameters
    ----------
    enabled : bool
        Whether to apply consolidation (default: False)
    priority_column : str, optional
        Column name for priority ranking (e.g., 'source_system', 'data_quality')
    priority_column_source : str
        Which dataframe the priority_column is from: 'df1' or 'df2' (default: 'df2')
    priority_order : List[str], optional
        Priority order for values in priority_column, e.g., ['system_a', 'system_b', 'system_c']
        First = highest priority
    priority_mode : str
        How to apply priority: 'strict' or 'threshold' (default: 'strict')
        - 'strict': Always prefer higher priority, regardless of score difference
        - 'threshold': Only use priority if similarity scores are within threshold
    priority_threshold : float
        For threshold mode: max score difference to apply priority (default: 10.0)
        Example: If threshold=10, priority only used when scores differ by ≤10 points
    metadata_columns : Dict[str, List[str]], optional
        Additional columns to include in consolidation, by source dataframe:
        {
            'df1': ['column1', 'column2'],
            'df2': ['column3', 'column4']
        }
    filter_function : Callable, optional
        Custom function to filter matches before consolidation.
        Function signature: func(row: pd.Series) -> bool
        Row contains: df1_id, df1_name, df2_id, df2_name, similarity_score, 
                     plus any metadata_columns
        Example: lambda row: row['source_system'] == 'system_a' and row['similarity_score'] > 85
    selection_function : Callable, optional
        Custom function to select main match from candidates.
        Function signature: func(matches_df: pd.DataFrame) -> pd.Series (single row)
        If not provided, uses priority_order + similarity_score based on priority_mode
        Example: lambda df: df.loc[df['similarity_score'].idxmax()]
    keep_all_matches : bool
        If True, keep all match IDs in 'all_match_ids' column (default: True)
    other_candidates_column : str
        Name for column with other candidate IDs (default: 'other_candidates')
    
    Examples
    --------
    Simple strict priority (always prefer system_a):
    >>> config = ConsolidationConfig(
    ...     enabled=True,
    ...     priority_column='source_system',
    ...     priority_order=['system_a', 'system_b'],
    ...     priority_mode='strict'
    ... )
    
    Threshold priority (only prefer system_a if scores are close):
    >>> config = ConsolidationConfig(
    ...     enabled=True,
    ...     priority_column='source_system',
    ...     priority_order=['system_a', 'system_b'],
    ...     priority_mode='threshold',
    ...     priority_threshold=10.0  # Only use priority if scores within 10 points
    ... )
    
    With score filter:
    >>> config = ConsolidationConfig(
    ...     enabled=True,
    ...     priority_column='source_system',
    ...     priority_order=['system_a', 'system_b'],
    ...     filter_function=lambda row: row['similarity_score'] > 85
    ... )
    
    With additional metadata and custom filter:
    >>> config = ConsolidationConfig(
    ...     enabled=True,
    ...     priority_column='source_system',
    ...     priority_order=['system_a', 'system_b'],
    ...     metadata_columns={
    ...         'df1': ['entity_type', 'country'],
    ...         'df2': ['source_system', 'status']
    ...     },
    ...     filter_function=lambda row: (
    ...         row['similarity_score'] > 80 and 
    ...         row['status'] == 'active'
    ...     )
    ... )
    
    Complex custom selection logic:
    >>> def custom_selector(matches_df):
    ...     # Custom logic: prefer verified sources if score > 90
    ...     verified = matches_df[matches_df['data_quality'] == 'verified']
    ...     if not verified.empty and verified['similarity_score'].max() > 90:
    ...         return verified.loc[verified['similarity_score'].idxmax()]
    ...     return matches_df.loc[matches_df['similarity_score'].idxmax()]
    >>> 
    >>> config = ConsolidationConfig(
    ...     enabled=True,
    ...     metadata_columns={'df2': ['data_quality']},
    ...     selection_function=custom_selector,
    ...     other_candidates_column='alternative_matches'
    ... )
    """
    
    enabled: bool = False
    priority_column: Optional[str] = None
    priority_column_source: str = 'df2'  # 'df1' or 'df2'
    priority_order: Optional[List[str]] = None
    priority_mode: str = 'strict'  # 'strict' or 'threshold'
    priority_threshold: float = 10.0  # For threshold mode
    metadata_columns: Optional[Dict[str, List[str]]] = None
    filter_function: Optional[Callable[[pd.Series], bool]] = None
    selection_function: Optional[Callable[[pd.DataFrame], pd.Series]] = None
    keep_all_matches: bool = True
    other_candidates_column: str = 'other_candidates'


class ConsolidationEngine:
    """Engine for consolidating cross-dataset matches based on rules"""
    
    def __init__(self, config: ConsolidationConfig):
        self.config = config
    
    def consolidate(
        self,
        cross_matches: pd.DataFrame,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        id_col1: str,
        id_col2: str,
        name_col2: str
    ) -> pd.DataFrame:
        """
        Apply consolidation rules to select main match for each df1 entity.
        
        Parameters
        ----------
        cross_matches : pd.DataFrame
            Dataframe with columns: df1_id, df1_name, df2_id, df2_name, similarity_score
        df1 : pd.DataFrame
            First dataset
        df2 : pd.DataFrame
            Second dataset
        id_col1 : str
            ID column in df1
        id_col2 : str
            ID column in df2
        name_col2 : str
            Name column in df2
            
        Returns
        -------
        pd.DataFrame
            Consolidated dataframe with one row per df1 entity
        """
        
        if not self.config.enabled:
            # No consolidation - return original df1
            return df1.copy()
        
        if cross_matches.empty:
            result = df1.copy()
            result[f'main_{id_col2}'] = None
            result[f'main_{name_col2}'] = None
            result['main_similarity_score'] = None
            if self.config.priority_column:
                result[f'main_{self.config.priority_column}'] = None
            result[self.config.other_candidates_column] = None
            result['total_matches'] = 0
            if self.config.keep_all_matches:
                result['all_match_ids'] = None
            return result
        
        # Enrich matches with metadata
        enriched_matches = self._enrich_matches(cross_matches, df1, df2, id_col1, id_col2)
        
        # Apply filter function if provided
        if self.config.filter_function:
            mask = enriched_matches.apply(self.config.filter_function, axis=1)
            enriched_matches = enriched_matches[mask].copy()
        
        # Consolidate matches for each df1 entity
        consolidated_rows = []
        
        for df1_id in df1[id_col1].unique():
            entity_matches = enriched_matches[enriched_matches['df1_id'] == df1_id].copy()
            
            # Get base df1 row
            df1_row = df1[df1[id_col1] == df1_id].iloc[0].to_dict()
            
            if entity_matches.empty:
                # No matches (possibly filtered out)
                df1_row[f'main_{id_col2}'] = None
                df1_row[f'main_{name_col2}'] = None
                df1_row['main_similarity_score'] = None
                if self.config.priority_column:
                    df1_row[f'main_{self.config.priority_column}'] = None
                df1_row[self.config.other_candidates_column] = None
                df1_row['total_matches'] = 0
                if self.config.keep_all_matches:
                    df1_row['all_match_ids'] = None
                consolidated_rows.append(df1_row)
                continue
            
            # Select main match
            main_match = self._select_main_match(entity_matches)
            other_matches = entity_matches[entity_matches.index != main_match.name]
            
            # Add match information
            df1_row[f'main_{id_col2}'] = main_match['df2_id']
            df1_row[f'main_{name_col2}'] = main_match['df2_name']
            df1_row['main_similarity_score'] = main_match['similarity_score']
            
            # Add priority column if specified
            if self.config.priority_column and self.config.priority_column in main_match:
                df1_row[f'main_{self.config.priority_column}'] = main_match[self.config.priority_column]
            
            # Other candidates
            if not other_matches.empty:
                other_ids = ', '.join(str(x) for x in other_matches['df2_id'].tolist())
                df1_row[self.config.other_candidates_column] = other_ids
            else:
                df1_row[self.config.other_candidates_column] = None
            
            df1_row['total_matches'] = len(entity_matches)
            
            # All match IDs
            if self.config.keep_all_matches:
                df1_row['all_match_ids'] = ', '.join(str(x) for x in entity_matches['df2_id'].tolist())
            
            consolidated_rows.append(df1_row)
        
        return pd.DataFrame(consolidated_rows)
    
    def _enrich_matches(
        self,
        matches: pd.DataFrame,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        id_col1: str,
        id_col2: str
    ) -> pd.DataFrame:
        """Add metadata columns to matches dataframe"""
        
        enriched = matches.copy()
        
        # Add df1 metadata if specified
        if self.config.metadata_columns and 'df1' in self.config.metadata_columns:
            df1_meta_cols = [id_col1] + self.config.metadata_columns['df1']
            df1_meta = df1[df1_meta_cols].copy()
            enriched = enriched.merge(
                df1_meta,
                left_on='df1_id',
                right_on=id_col1,
                how='left',
                suffixes=('', '_df1_meta')
            )
        
        # Add df2 metadata if specified
        if self.config.metadata_columns and 'df2' in self.config.metadata_columns:
            df2_meta_cols = [id_col2] + self.config.metadata_columns['df2']
            df2_meta = df2[df2_meta_cols].copy()
            enriched = enriched.merge(
                df2_meta,
                left_on='df2_id',
                right_on=id_col2,
                how='left',
                suffixes=('', '_df2_meta')
            )
        
        # Add priority column if specified
        if self.config.priority_column:
            source_df = df2 if self.config.priority_column_source == 'df2' else df1
            id_col = id_col2 if self.config.priority_column_source == 'df2' else id_col1
            id_match = 'df2_id' if self.config.priority_column_source == 'df2' else 'df1_id'
            
            if self.config.priority_column in source_df.columns:
                priority_data = source_df[[id_col, self.config.priority_column]].copy()
                enriched = enriched.merge(
                    priority_data,
                    left_on=id_match,
                    right_on=id_col,
                    how='left',
                    suffixes=('', '_priority')
                )
        
        return enriched
    
    def _select_main_match(self, entity_matches: pd.DataFrame) -> pd.Series:
        """Select the main match from candidates"""
        
        # Use custom selection function if provided
        if self.config.selection_function:
            return self.config.selection_function(entity_matches)
        
        # Use priority_order + similarity_score based on priority mode
        if self.config.priority_order and self.config.priority_column:
            # Create priority ranking
            priority_map = {source: idx for idx, source in enumerate(self.config.priority_order)}
            entity_matches = entity_matches.copy()
            entity_matches['priority_rank'] = entity_matches[self.config.priority_column].map(
                lambda x: priority_map.get(x, len(self.config.priority_order))
            )
            
            if self.config.priority_mode == 'strict':
                # STRICT: Always prefer higher priority, regardless of score difference
                entity_matches = entity_matches.sort_values(
                    ['priority_rank', 'similarity_score'],
                    ascending=[True, False]
                )
                return entity_matches.iloc[0]
            
            elif self.config.priority_mode == 'threshold':
                # THRESHOLD: Only use priority if scores are within threshold
                # First, find the best match by score alone
                best_by_score = entity_matches.loc[entity_matches['similarity_score'].idxmax()]
                best_score = best_by_score['similarity_score']
                
                # Filter matches within threshold of best score
                within_threshold = entity_matches[
                    (best_score - entity_matches['similarity_score']) <= self.config.priority_threshold
                ].copy()
                
                if len(within_threshold) > 1:
                    # Multiple matches within threshold - use priority
                    within_threshold = within_threshold.sort_values(
                        ['priority_rank', 'similarity_score'],
                        ascending=[True, False]
                    )
                    return within_threshold.iloc[0]
                else:
                    # Only one match within threshold (or threshold is 0) - return best by score
                    return best_by_score
            
            else:
                raise ValueError(f"Invalid priority_mode: {self.config.priority_mode}. Use 'strict' or 'threshold'")
        
    
        return entity_matches.loc[entity_matches['similarity_score'].idxmax()]