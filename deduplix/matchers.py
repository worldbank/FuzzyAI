
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz, process
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
from .core import Matcher, MatchResult


class FuzzyMatcher(Matcher):
    """
    Fuzzy string matching using RapidFuzz with preprocessing support.
    
    Parameters
    ----------
    threshold : float, default=80.0
        Minimum similarity score (0-100) for matches
    scorer : str, default='ratio'
        Scoring algorithm: 'ratio', 'partial_ratio', 'token_sort_ratio',
        'token_set_ratio', or 'WRatio'
    max_matches_per_entity : int or None, default=None
        Maximum matches per entity. None for unlimited matches
    n_workers : int or None, default=None
        Number of parallel workers. None uses min(cpu_count-1, 4)
    use_multiprocessing : bool, default=False
        Whether to use multiprocessing (disabled on Windows)
    lowercase : bool, default=True
        Convert text to lowercase during preprocessing
    strip_whitespace : bool, default=True
        Remove extra whitespace during preprocessing
    remove_punctuation : bool, default=False
        Remove punctuation during preprocessing
    punctuation_pattern : str, default=r'[^\w\s]'
        Regex pattern for punctuation removal
    """
    
    def __init__(
        self,
        threshold: float = 80.0,
        scorer: str = 'ratio',
        max_matches_per_entity: Optional[int] = None,
        n_workers: Optional[int] = None,
        use_multiprocessing: bool = False,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        remove_punctuation: bool = False,
        punctuation_pattern: str = r'[^\w\s]'
    ):
        self.threshold = threshold
        self.scorer_name = scorer
        self.scorer = self._get_scorer(scorer)
        self.max_matches = max_matches_per_entity
        self.n_workers = n_workers or min(mp.cpu_count() - 1, 4)
        
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace
        self.remove_punctuation = remove_punctuation
        self.punctuation_pattern = punctuation_pattern
        
        import platform
        self.use_multiprocessing = use_multiprocessing and platform.system() != 'Windows'
    
    def _get_scorer(self, scorer_name: str):
        """
        Get scorer function from name.
        
        Parameters
        ----------
        scorer_name : str
            Name of the scoring algorithm
            
        Returns
        -------
        callable
            RapidFuzz scorer function
        """
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
        Preprocess text according to configured settings.
        
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
        
        if self.strip_whitespace:
            processed = processed.strip()
            processed = re.sub(r'\s+', ' ', processed)
        
        if self.lowercase:
            processed = processed.lower()
        
        if self.remove_punctuation:
            processed = re.sub(self.punctuation_pattern, '', processed)
            if self.strip_whitespace:
                processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Prepare dataframe with vectorized preprocessing.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with 'id' and 'name' columns
            
        Returns
        -------
        tuple
            (df_prep, prep_to_ids, prep_to_original) where:
            - df_prep: DataFrame with 'name_preprocessed' column added
            - prep_to_ids: Dict mapping preprocessed names to list of IDs
            - prep_to_original: Dict mapping preprocessed names to original names
        """
        df_prep = df.copy()
        df_prep['name_preprocessed'] = df_prep['name'].apply(self.preprocess_text)
        
        valid_mask = df_prep['name_preprocessed'].astype(bool)
        df_valid = df_prep[valid_mask]
        
        grouped = df_valid.groupby('name_preprocessed', sort=False)
        
        prep_to_ids = grouped['id'].apply(list).to_dict()
        prep_to_original = grouped['name'].first().to_dict()
        
        return df_prep, prep_to_ids, prep_to_original
    
    def _process_batch(self, args: Tuple) -> List[Dict]:
        """
        Process a batch of entities for matching.
        
        Parameters
        ----------
        args : tuple
            (batch_df, all_preprocessed_names, prep_to_ids, prep_to_original)
            
        Returns
        -------
        list of dict
            Match results with keys: id1, id2, name1, name2, similarity_score
        """
        batch_df, all_preprocessed_names, prep_to_ids, prep_to_original = args
        results = []
        
        batch_records = batch_df.to_dict('records')
        
        for row in batch_records:
            entity_id = row['id']
            entity_name_original = row['name']
            entity_name_preprocessed = row['name_preprocessed']
            
            if not entity_name_preprocessed:
                continue
            
            matches = process.extract(
                entity_name_preprocessed,
                all_preprocessed_names,
                scorer=self.scorer,
                score_cutoff=self.threshold
            )
            
            match_count = 0
            for match_name_prep, score, _ in matches:
                match_ids = prep_to_ids.get(match_name_prep, [])
                match_name_original = prep_to_original.get(match_name_prep, match_name_prep)
                
                for match_id in match_ids:
                    if match_id != entity_id:
                        results.append({
                            'id1': entity_id,
                            'id2': match_id,
                            'name1': entity_name_original,
                            'name2': match_name_original,
                            'similarity_score': score
                        })
                        match_count += 1
                        
                        if self.max_matches is not None and match_count >= self.max_matches:
                            break
                
                if self.max_matches is not None and match_count >= self.max_matches:
                    break
        
        return results
    
    def _remove_symmetric_pairs(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove symmetric duplicate pairs using set-based O(n) approach.
        
        Parameters
        ----------
        pairs_df : pd.DataFrame
            DataFrame with id1 and id2 columns
            
        Returns
        -------
        pd.DataFrame
            DataFrame with symmetric pairs removed
        """
        if pairs_df.empty:
            return pairs_df
        
        seen = set()
        keep_indices = []
        
        for idx, (id1, id2) in enumerate(zip(pairs_df['id1'], pairs_df['id2'])):
            key = (min(id1, id2), max(id1, id2)) if id1 < id2 else (id2, id1)
            if key not in seen:
                seen.add(key)
                keep_indices.append(idx)
        
        return pairs_df.iloc[keep_indices]
    
    def find_matches(self, df: pd.DataFrame, **kwargs) -> MatchResult:
        """
        Find fuzzy matches within a single dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'id' and 'name' columns
        **kwargs
            Additional keyword arguments (unused)
            
        Returns
        -------
        MatchResult
            Object containing matched pairs and metadata
        """
        df_clean = df.dropna(subset=['name']).copy()
        
        if df_clean.empty:
            return MatchResult(
                pairs=pd.DataFrame(columns=['id1', 'id2', 'name1', 'name2', 'similarity_score']),
                metadata={'message': 'No valid entities to match'}
            )
        
        df_prep, prep_to_ids, prep_to_original = self._prepare_dataframe(df_clean)
        all_preprocessed_names = list(prep_to_ids.keys())
        
        batch_size = max(1, len(df_prep) // (self.n_workers * 4))
        batches = []
        
        for i in range(0, len(df_prep), batch_size):
            batch_df = df_prep.iloc[i:i+batch_size]
            batches.append((batch_df, all_preprocessed_names, prep_to_ids, prep_to_original))
        
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self._process_batch, batch) for batch in batches]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fuzzy matching"):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"Error in batch processing: {e}")
        
        if all_results:
            pairs_df = pd.DataFrame(all_results)
            pairs_df = self._remove_symmetric_pairs(pairs_df)
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
    
    def _process_cross_batch(self, args: Tuple) -> List[Dict]:
        """
        Process a batch of entities for cross-dataset matching.
        
        Parameters
        ----------
        args : tuple
            (batch_df, df2_preprocessed_names, prep_to_ids2, prep_to_original2)
            
        Returns
        -------
        list of dict
            Match results with keys: id1, id2, name1, name2, similarity_score,
            source1, source2
        """
        batch_df, df2_preprocessed_names, prep_to_ids2, prep_to_original2 = args
        results = []
        
        batch_records = batch_df.to_dict('records')
        
        for row in batch_records:
            entity_id1 = row['id']
            entity_name1_original = row['name']
            entity_name1_preprocessed = row['name_preprocessed']
            
            if not entity_name1_preprocessed:
                continue
            
            matches = process.extract(
                entity_name1_preprocessed,
                df2_preprocessed_names,
                scorer=self.scorer,
                score_cutoff=self.threshold
            )
            
            match_count = 0
            for match_name_prep, score, _ in matches:
                match_ids2 = prep_to_ids2.get(match_name_prep, [])
                match_name2_original = prep_to_original2.get(match_name_prep, match_name_prep)
                
                for match_id2 in match_ids2:
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
                    
                    if self.max_matches is not None and match_count >= self.max_matches:
                        break
                
                if self.max_matches is not None and match_count >= self.max_matches:
                    break
        
        return results
    
    def _find_cross_matches_multi_column(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        id_column1: str,
        id_column2: str,
        name_columns1: List[str],
        name_columns2: List[str]
    ) -> MatchResult:
        """
        Find cross-dataset matches using multiple name column combinations.
        
        Caches df1 and df2 preparations to avoid redundant preprocessing when
        trying multiple column combinations.
        
        Parameters
        ----------
        df1 : pd.DataFrame
            First dataset
        df2 : pd.DataFrame
            Second dataset
        id_column1 : str
            ID column name in df1
        id_column2 : str
            ID column name in df2
        name_columns1 : list of str
            Name columns to try in df1
        name_columns2 : list of str
            Name columns to try in df2
            
        Returns
        -------
        MatchResult
            Best matches across all column combinations
        """
        all_matches = []
        total_combinations = len(name_columns1) * len(name_columns2)
        
        print(f"  Trying {total_combinations} column combinations...")
        
        df1_cache = {}
        for col1 in name_columns1:
            if col1 not in df1.columns:
                print(f"  Warning: Column '{col1}' not found in df1, skipping")
                continue
            df1_subset = df1[[id_column1, col1]].dropna().copy()
            if df1_subset.empty:
                continue
            df1_subset = df1_subset.rename(columns={id_column1: 'id', col1: 'name'})
            df1_cache[col1] = self._prepare_dataframe(df1_subset)
        
        df2_cache = {}
        for col2 in name_columns2:
            if col2 not in df2.columns:
                print(f"  Warning: Column '{col2}' not found in df2, skipping")
                continue
            df2_subset = df2[[id_column2, col2]].dropna().copy()
            if df2_subset.empty:
                continue
            df2_subset = df2_subset.rename(columns={id_column2: 'id', col2: 'name'})
            df2_cache[col2] = self._prepare_dataframe(df2_subset)
        
        for col1 in name_columns1:
            if col1 not in df1_cache:
                continue
                
            df1_prep, prep_to_ids1, prep_to_original1 = df1_cache[col1]
            
            for col2 in name_columns2:
                if col2 not in df2_cache:
                    continue
                
                df2_prep, prep_to_ids2, prep_to_original2 = df2_cache[col2]
                df2_preprocessed_names = list(prep_to_ids2.keys())
                
                batch_size = max(1, len(df1_prep) // (self.n_workers * 4))
                batches = []
                
                for i in range(0, len(df1_prep), batch_size):
                    batch_df = df1_prep.iloc[i:i+batch_size]
                    batches.append((batch_df, df2_preprocessed_names, prep_to_ids2, prep_to_original2))
                
                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = [executor.submit(self._process_cross_batch, batch) for batch in batches]
                    
                    for future in as_completed(futures):
                        try:
                            batch_results = future.result()
                            for match in batch_results:
                                match['matched_column1'] = col1
                                match['matched_column2'] = col2
                            all_matches.extend(batch_results)
                        except Exception as e:
                            print(f"  Error in batch: {e}")
        
        if not all_matches:
            return MatchResult(
                pairs=pd.DataFrame(columns=[
                    'id1', 'id2', 'name1', 'name2', 'similarity_score', 
                    'source1', 'source2', 'matched_column1', 'matched_column2'
                ]),
                metadata={'message': 'No matches found across any column combinations'}
            )
        
        matches_df = pd.DataFrame(all_matches)
        
        id1_str = matches_df['id1'].astype(str).values
        id2_str = matches_df['id2'].astype(str).values
        matches_df['pair_key'] = [
            (a, b) if a <= b else (b, a) 
            for a, b in zip(id1_str, id2_str)
        ]
        
        best_matches = matches_df.loc[
            matches_df.groupby('pair_key')['similarity_score'].idxmax()
        ].copy()
        
        best_matches = best_matches.drop(columns=['pair_key'])
        best_matches = best_matches.sort_values('similarity_score', ascending=False)
        
        print(f"  Found {len(all_matches)} total matches across combinations")
        print(f"  Consolidated to {len(best_matches)} unique entity pairs (keeping best match)")
        
        metadata = {
            'threshold': self.threshold,
            'scorer': self.scorer_name,
            'df1_entities': len(df1),
            'df2_entities': len(df2),
            'cross_matches_found': len(best_matches),
            'matching_type': 'cross_dataset_multi_column',
            'name_columns1': name_columns1,
            'name_columns2': name_columns2,
            'total_combinations_tried': len(name_columns1) * len(name_columns2),
            'preprocessing': {
                'lowercase': self.lowercase,
                'strip_whitespace': self.strip_whitespace,
                'remove_punctuation': self.remove_punctuation
            }
        }
        
        return MatchResult(pairs=best_matches, metadata=metadata)
    
    def find_cross_matches(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame,
        id_column1: str = 'id',
        name_column1: str = 'name', 
        id_column2: str = 'id',
        name_column2: str = 'name',
        name_columns1: Optional[List[str]] = None,
        name_columns2: Optional[List[str]] = None,
        **kwargs
    ) -> MatchResult:
        """
        Find matches between two datasets with optional multiple name columns.
        
        Parameters
        ----------
        df1 : pd.DataFrame
            First dataset
        df2 : pd.DataFrame
            Second dataset
        id_column1 : str, default='id'
            ID column name in df1
        name_column1 : str, default='name'
            Primary name column in df1
        id_column2 : str, default='id'
            ID column name in df2
        name_column2 : str, default='name'
            Primary name column in df2
        name_columns1 : list of str or None, optional
            Multiple name columns to try in df1. If provided, will try all
            combinations with name_columns2 and keep best matches
        name_columns2 : list of str or None, optional
            Multiple name columns to try in df2
        **kwargs
            Additional keyword arguments (unused)
            
        Returns
        -------
        MatchResult
            Cross-dataset matches with similarity scores
        """
        cols1 = name_columns1 if name_columns1 else [name_column1]
        cols2 = name_columns2 if name_columns2 else [name_column2]
        
        if name_columns1 or name_columns2:
            return self._find_cross_matches_multi_column(
                df1, df2, id_column1, id_column2, cols1, cols2
            )
        
        df1_clean = df1[[id_column1, name_column1]].dropna().copy()
        df2_clean = df2[[id_column2, name_column2]].dropna().copy()
        
        df1_clean = df1_clean.rename(columns={id_column1: 'id', name_column1: 'name'})
        df2_clean = df2_clean.rename(columns={id_column2: 'id', name_column2: 'name'})
        
        if df1_clean.empty or df2_clean.empty:
            return MatchResult(
                pairs=pd.DataFrame(columns=['id1', 'id2', 'name1', 'name2', 'similarity_score', 'source1', 'source2']),
                metadata={'message': 'One or both datasets empty'}
            )
        
        df1_prep, prep_to_ids1, prep_to_original1 = self._prepare_dataframe(df1_clean)
        df2_prep, prep_to_ids2, prep_to_original2 = self._prepare_dataframe(df2_clean)
        
        df2_preprocessed_names = list(prep_to_ids2.keys())
        
        batch_size = max(1, len(df1_prep) // (self.n_workers * 4))
        batches = []
        
        for i in range(0, len(df1_prep), batch_size):
            batch_df = df1_prep.iloc[i:i+batch_size]
            batches.append((batch_df, df2_preprocessed_names, prep_to_ids2, prep_to_original2))
        
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self._process_cross_batch, batch) for batch in batches]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Cross-dataset matching"):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    print(f"Error in cross-dataset batch processing: {e}")
        
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