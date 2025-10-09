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