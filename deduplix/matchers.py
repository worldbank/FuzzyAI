

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz, process
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
from .core import Matcher, MatchResult


class FuzzyMatcher(Matcher):
    """Fuzzy string matching using RapidFuzz with preprocessing support"""
    
    def __init__(
        self,
        threshold: float = 80.0,
        scorer: str = 'ratio',
        max_matches_per_entity: int = 100,
        n_workers: Optional[int] = None,
        use_multiprocessing: bool = False,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        remove_punctuation: bool = False,
        punctuation_pattern: str = r'[^\w\s]'
    ):
        """
        Initialize FuzzyMatcher with preprocessing options.
        
        Parameters
        ----------
        threshold : float
            Minimum similarity score (0-100) to consider a match
        scorer : str
            RapidFuzz scorer name: 'ratio', 'partial_ratio', 'token_sort_ratio', 
            'token_set_ratio', 'WRatio'
        max_matches_per_entity : int
            Maximum number of matches to return per entity
        n_workers : int, optional
            Number of worker threads (default: CPU count - 1, max 4)
        use_multiprocessing : bool
            Use multiprocessing instead of threading (disabled on Windows)
        lowercase : bool
            Convert all text to lowercase before matching
        strip_whitespace : bool
            Remove leading/trailing whitespace and normalize internal spaces
        remove_punctuation : bool
            Remove punctuation before matching
        punctuation_pattern : str
            Regex pattern for punctuation removal (default: removes non-alphanumeric)
        """
        self.threshold = threshold
        self.scorer_name = scorer
        self.scorer = self._get_scorer(scorer)
        self.max_matches = max_matches_per_entity
        self.n_workers = n_workers or min(mp.cpu_count() - 1, 4)
        
        # Preprocessing options
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace
        self.remove_punctuation = remove_punctuation
        self.punctuation_pattern = punctuation_pattern
        
        # Platform-specific multiprocessing
        import platform
        self.use_multiprocessing = use_multiprocessing and platform.system() != 'Windows'
    
    def _get_scorer(self, scorer_name: str):
        """Get scorer function from name"""
        scorers = {
            'ratio': fuzz.ratio,
            'partial_ratio': fuzz.partial_ratio,
            'token_sort_ratio': fuzz.token_sort_ratio,
            'token_set_ratio': fuzz.token_set_ratio,
            'WRatio': fuzz.WRatio
        }
        return scorers.get(scorer_name, fuzz.ratio)
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply preprocessing to a single text string.
        
        Parameters
        ----------
        text : str
            Input text to preprocess
            
        Returns
        -------
        str
            Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        processed = str(text)
        
        # Strip whitespace
        if self.strip_whitespace:
            processed = processed.strip()
            # Normalize internal whitespace (multiple spaces -> single space)
            processed = re.sub(r'\s+', ' ', processed)
        
        # Lowercase
        if self.lowercase:
            processed = processed.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            processed = re.sub(self.punctuation_pattern, '', processed)
            # Clean up any extra spaces created by punctuation removal
            if self.strip_whitespace:
                processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
        """
        Prepare dataframe with preprocessed names while preserving originals.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with 'id' and 'name' columns
            
        Returns
        -------
        tuple[pd.DataFrame, Dict[str, str]]
            (DataFrame with preprocessed names, mapping from preprocessed -> original name)
        """
        df_prep = df.copy()
        
        # Apply preprocessing
        df_prep['name_preprocessed'] = df_prep['name'].apply(self.preprocess_text)
        
        # Create mapping from preprocessed name back to original
        # Handle multiple entities with same preprocessed name by keeping first original
        preprocessed_to_original = {}
        for _, row in df_prep.iterrows():
            prep_name = row['name_preprocessed']
            if prep_name and prep_name not in preprocessed_to_original:
                preprocessed_to_original[prep_name] = row['name']
        
        return df_prep, preprocessed_to_original
    
    def _process_batch(self, args):
        """Process a batch of entities for matching"""
        batch_df, all_preprocessed_names, prep_to_id, prep_to_original = args
        results = []
        
        for _, row in batch_df.iterrows():
            entity_id = row['id']
            entity_name_original = row['name']
            entity_name_preprocessed = row['name_preprocessed']
            
            if not entity_name_preprocessed:
                continue
            
            # Find matches using RapidFuzz on preprocessed names
            matches = process.extract(
                entity_name_preprocessed,
                all_preprocessed_names,
                scorer=self.scorer,
                limit=None
            )
            
            # Filter by threshold and exclude self-matches
            match_count = 0
            for match_name_prep, score, _ in matches:
                if score >= self.threshold:
                    match_id = prep_to_id.get(match_name_prep)
                    if match_id and match_id != entity_id:
                        # Use original names in results for readability
                        match_name_original = prep_to_original.get(match_name_prep, match_name_prep)
                        
                        results.append({
                            'id1': entity_id,
                            'id2': match_id,
                            'name1': entity_name_original,
                            'name2': match_name_original,
                            'similarity_score': score
                        })
                        match_count += 1
                        
                        if match_count >= self.max_matches:
                            break
        
        return results
    
    def find_matches(self, df: pd.DataFrame, **kwargs) -> MatchResult:
        """
        Find potential duplicate pairs using fuzzy matching with preprocessing.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with 'id' and 'name' columns
        **kwargs
            Additional parameters (currently unused)
            
        Returns
        -------
        MatchResult
            Result object containing matched pairs and metadata
        """
        # Prepare data with preprocessing
        df_clean = df.dropna(subset=['name']).copy()
        
        if df_clean.empty:
            return MatchResult(
                pairs=pd.DataFrame(columns=['id1', 'id2', 'name1', 'name2', 'similarity_score']),
                metadata={'message': 'No valid entities to match'}
            )
        
        # Preprocess names
        df_prep, prep_to_original = self._prepare_dataframe(df_clean)
        
        # Get unique preprocessed names and create mappings
        all_preprocessed_names = df_prep['name_preprocessed'].unique().tolist()
        prep_to_id = dict(zip(df_prep['name_preprocessed'], df_prep['id']))
        
        # Split data into batches
        batch_size = max(1, len(df_prep) // (self.n_workers * 4))
        batches = []
        
        for i in range(0, len(df_prep), batch_size):
            batch_df = df_prep.iloc[i:i+batch_size]
            batches.append((batch_df, all_preprocessed_names, prep_to_id, prep_to_original))
        
        # Process batches in parallel
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self._process_batch, batch) for batch in batches]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fuzzy matching"):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"Error in batch processing: {e}")
        
        # Create results dataframe
        if all_results:
            pairs_df = pd.DataFrame(all_results)
            # Remove symmetric duplicates: keep only (id1, id2) where id1 < id2
            pairs_df['min_id'] = pairs_df[['id1', 'id2']].min(axis=1)
            pairs_df['max_id'] = pairs_df[['id1', 'id2']].max(axis=1)
            pairs_df = pairs_df.drop_duplicates(subset=['min_id', 'max_id'])
            pairs_df = pairs_df.drop(columns=['min_id', 'max_id'])
            pairs_df = pairs_df.sort_values('similarity_score', ascending=False)
        else:
            pairs_df = pd.DataFrame(columns=['id1', 'id2', 'name1', 'name2', 'similarity_score'])
        
        metadata = {
            'threshold': self.threshold,
            'scorer': self.scorer_name,
            'total_entities': len(df),
            'matches_found': len(pairs_df),
            'preprocessing': {
                'lowercase': self.lowercase,
                'strip_whitespace': self.strip_whitespace,
                'remove_punctuation': self.remove_punctuation
            }
        }
        
        return MatchResult(pairs=pairs_df, metadata=metadata)
    
    def _process_cross_batch(self, args):
        """Process a batch of df1 entities against all df2 entities"""
        batch_df, df2_preprocessed_names, prep_to_id2, prep_to_original2 = args
        results = []
        
        for _, row in batch_df.iterrows():
            entity_id1 = row['id']
            entity_name1_original = row['name']
            entity_name1_preprocessed = row['name_preprocessed']
            
            if not entity_name1_preprocessed:
                continue
            
            # Find matches in df2 using preprocessed names
            matches = process.extract(
                entity_name1_preprocessed,
                df2_preprocessed_names,
                scorer=self.scorer,
                limit=None
            )
            
            # Filter by threshold
            match_count = 0
            for match_name_prep, score, _ in matches:
                if score >= self.threshold:
                    match_id2 = prep_to_id2.get(match_name_prep)
                    if match_id2:
                        # Use original names in results
                        match_name2_original = prep_to_original2.get(match_name_prep, match_name_prep)
                        
                        results.append({
                            'id1': entity_id1,
                            'id2': match_id2,
                            'name1': entity_name1_original,
                            'name2': match_name2_original,
                            'similarity_score': score,
                            'source1': 'df1',
                            'source2': 'df2'
                        })
                        match_count += 1
                        
                        if match_count >= self.max_matches:
                            break
        
        return results
    
    def find_cross_matches(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame,
        id_column1: str = 'id',
        name_column1: str = 'name', 
        id_column2: str = 'id',
        name_column2: str = 'name',
        **kwargs
    ) -> MatchResult:
        """
        Find matches between two different dataframes with preprocessing.
        
        Parameters
        ----------
        df1 : pd.DataFrame
            First dataframe
        df2 : pd.DataFrame
            Second dataframe
        id_column1 : str
            ID column name in df1
        name_column1 : str
            Name column name in df1
        id_column2 : str
            ID column name in df2
        name_column2 : str
            Name column name in df2
        **kwargs
            Additional parameters (currently unused)
            
        Returns
        -------
        MatchResult
            Result object containing cross-matched pairs and metadata
        """
        # Prepare data
        df1_clean = df1[[id_column1, name_column1]].dropna().copy()
        df2_clean = df2[[id_column2, name_column2]].dropna().copy()
        
        # Standardize column names
        df1_clean = df1_clean.rename(columns={id_column1: 'id', name_column1: 'name'})
        df2_clean = df2_clean.rename(columns={id_column2: 'id', name_column2: 'name'})
        
        if df1_clean.empty or df2_clean.empty:
            return MatchResult(
                pairs=pd.DataFrame(columns=['id1', 'id2', 'name1', 'name2', 'similarity_score', 'source1', 'source2']),
                metadata={'message': 'One or both datasets empty'}
            )
        
        # Preprocess both dataframes
        df1_prep, prep_to_original1 = self._prepare_dataframe(df1_clean)
        df2_prep, prep_to_original2 = self._prepare_dataframe(df2_clean)
        
        # Get unique preprocessed names and create mappings for df2
        df2_preprocessed_names = df2_prep['name_preprocessed'].unique().tolist()
        prep_to_id2 = dict(zip(df2_prep['name_preprocessed'], df2_prep['id']))
        
        # Split df1 into batches for parallel processing
        batch_size = max(1, len(df1_prep) // (self.n_workers * 4))
        batches = []
        
        for i in range(0, len(df1_prep), batch_size):
            batch_df = df1_prep.iloc[i:i+batch_size]
            batches.append((batch_df, df2_preprocessed_names, prep_to_id2, prep_to_original2))
        
        # Process batches with progress bar
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self._process_cross_batch, batch) for batch in batches]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Cross-dataset matching"):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"Error in cross-dataset batch processing: {e}")
        
        # Create results dataframe
        if all_results:
            pairs_df = pd.DataFrame(all_results)
            pairs_df = pairs_df.sort_values('similarity_score', ascending=False)
        else:
            pairs_df = pd.DataFrame(columns=['id1', 'id2', 'name1', 'name2', 'similarity_score', 'source1', 'source2'])
        
        metadata = {
            'threshold': self.threshold,
            'scorer': self.scorer_name,
            'df1_entities': len(df1_clean),
            'df2_entities': len(df2_clean),
            'cross_matches_found': len(pairs_df),
            'matching_type': 'cross_dataset',
            'preprocessing': {
                'lowercase': self.lowercase,
                'strip_whitespace': self.strip_whitespace,
                'remove_punctuation': self.remove_punctuation
            }
        }
        
        return MatchResult(pairs=pairs_df, metadata=metadata)