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
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> tuple:
        df_prep = df.copy()
        df_prep['name_preprocessed'] = df_prep['name'].apply(self.preprocess_text)
        
        prep_to_ids = {}
        prep_to_original = {}
        
        for _, row in df_prep.iterrows():
            prep_name = row['name_preprocessed']
            if prep_name:
                if prep_name not in prep_to_ids:
                    prep_to_ids[prep_name] = []
                    prep_to_original[prep_name] = row['name']
                prep_to_ids[prep_name].append(row['id'])
        
        return df_prep, prep_to_ids, prep_to_original
    
    def _process_batch(self, args):
        batch_df, all_preprocessed_names, prep_to_ids, prep_to_original = args
        results = []
        
        for _, row in batch_df.iterrows():
            entity_id = row['id']
            entity_name_original = row['name']
            entity_name_preprocessed = row['name_preprocessed']
            
            if not entity_name_preprocessed:
                continue
            
            matches = process.extract(
                entity_name_preprocessed,
                all_preprocessed_names,
                scorer=self.scorer,
                limit=None
            )
            
            match_count = 0
            for match_name_prep, score, _ in matches:
                if score >= self.threshold:
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
                            
                            if match_count >= self.max_matches:
                                break
                    
                    if match_count >= self.max_matches:
                        break
        
        return results
    
    def find_matches(self, df: pd.DataFrame, **kwargs) -> MatchResult:
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
        batch_df, df2_preprocessed_names, prep_to_ids2, prep_to_original2 = args
        results = []
        
        for _, row in batch_df.iterrows():
            entity_id1 = row['id']
            entity_name1_original = row['name']
            entity_name1_preprocessed = row['name_preprocessed']
            
            if not entity_name1_preprocessed:
                continue
            
            matches = process.extract(
                entity_name1_preprocessed,
                df2_preprocessed_names,
                scorer=self.scorer,
                limit=None
            )
            
            match_count = 0
            for match_name_prep, score, _ in matches:
                if score >= self.threshold:
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
                        
                        if match_count >= self.max_matches:
                            break
                    
                    if match_count >= self.max_matches:
                        break
        
        return results
    
<<<<<<< HEAD
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
        Internal method to handle matching across multiple column combinations.
        For each entity pair, keeps the best match across all column combinations.
        """
        
        all_matches = []
        total_combinations = len(name_columns1) * len(name_columns2)
        
        print(f"  Trying {total_combinations} column combinations...")
        
        # Try each combination of name columns
        for col1 in name_columns1:
            for col2 in name_columns2:
                if col1 not in df1.columns:
                    print(f"  Warning: Column '{col1}' not found in df1, skipping")
                    continue
                if col2 not in df2.columns:
                    print(f"  Warning: Column '{col2}' not found in df2, skipping")
                    continue
                
                # Prepare data for this column combination
                df1_subset = df1[[id_column1, col1]].dropna().copy()
                df2_subset = df2[[id_column2, col2]].dropna().copy()
                
                if df1_subset.empty or df2_subset.empty:
                    continue
                
                df1_subset = df1_subset.rename(columns={id_column1: 'id', col1: 'name'})
                df2_subset = df2_subset.rename(columns={id_column2: 'id', col2: 'name'})
                
                # Match on this combination
                df1_prep, prep_to_ids1, prep_to_original1 = self._prepare_dataframe(df1_subset)
                df2_prep, prep_to_ids2, prep_to_original2 = self._prepare_dataframe(df2_subset)
                df2_preprocessed_names = list(prep_to_ids2.keys())
                
                # Process batches for this combination
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
                            # Add column info to each match
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
        
        # Convert to DataFrame
        matches_df = pd.DataFrame(all_matches)
        
        # For each entity pair, keep only the best match across all column combinations
        # Convert IDs to strings to handle mixed types (int and str)
        matches_df['pair_key'] = matches_df.apply(
            lambda row: tuple(sorted([str(row['id1']), str(row['id2'])])), 
            axis=1
        )
        
        # Keep best match for each pair
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
    
=======
>>>>>>> c032314305f8e2afa96c70afb0030f76ad8c3a64
    def find_cross_matches(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame,
        id_column1: str = 'id',
        name_column1: str = 'name', 
        id_column2: str = 'id',
        name_column2: str = 'name',
<<<<<<< HEAD
        name_columns1: Optional[List[str]] = None,  # NEW: Multiple name columns for df1
        name_columns2: Optional[List[str]] = None,  # NEW: Multiple name columns for df2
        **kwargs
    ) -> MatchResult:
        """
        Find matches between two datasets with support for multiple name columns.
        
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
            If provided, will try matching on all combinations and keep best match
        name_columns2 : List[str], optional
            Multiple name columns to try for df2
        
        Returns
        -------
        MatchResult
            Cross-dataset matches with 'matched_column1' and 'matched_column2' 
            showing which columns produced the match
        
        Examples
        --------
        # Match using multiple columns
        >>> matcher.find_cross_matches(
        ...     df1, df2,
        ...     id_column1='id', 
        ...     name_columns1=['legal_name', 'short_name'],
        ...     id_column2='id',
        ...     name_columns2=['company_name', 'dba_name']
        ... )
        """
        
        # Use name_columns if provided, otherwise use single name_column
        cols1 = name_columns1 if name_columns1 else [name_column1]
        cols2 = name_columns2 if name_columns2 else [name_column2]
        
        # If using multiple columns, find matches for each combination
        if name_columns1 or name_columns2:
            return self._find_cross_matches_multi_column(
                df1, df2, id_column1, id_column2, cols1, cols2
            )
        
        # Original single-column logic
=======
        **kwargs
    ) -> MatchResult:
>>>>>>> c032314305f8e2afa96c70afb0030f76ad8c3a64
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