from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
from .core import Matcher, MatchResult


def _process_batch_worker(args):
    """Worker function for parallel processing (must be at module level for pickling)"""
    batch_df, all_names, name_to_id, scorer, threshold, max_matches = args
    results = []
    
    for _, row in batch_df.iterrows():
        entity_id = row['id']
        entity_name = row['name']
        
        if pd.isna(entity_name):
            continue
        
        # Find matches using RapidFuzz
        matches = process.extract(
            entity_name,
            all_names,
            scorer=scorer,
            limit=None
        )
        
        # Filter by threshold and exclude self-matches
        match_count = 0
        for match_name, score, _ in matches:
            if score >= threshold:
                match_id = name_to_id.get(match_name)
                if match_id and match_id != entity_id:
                    results.append({
                        'id1': entity_id,
                        'id2': match_id,
                        'name1': entity_name,
                        'name2': match_name,
                        'similarity_score': score
                    })
                    match_count += 1
                    
                    if match_count >= max_matches:
                        break
    
    return results


# deduplix/matchers.py
"""Fuzzy matching implementations"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
from .core import Matcher, MatchResult


class FuzzyMatcher(Matcher):
    """Fuzzy string matching using RapidFuzz"""
    
    def __init__(
        self,
        threshold: float = 80.0,
        scorer: str = 'ratio',
        max_matches_per_entity: int = 100,
        n_workers: Optional[int] = None,
        use_multiprocessing: bool = False  # Default to False on Windows
    ):
        self.threshold = threshold
        self.scorer_name = scorer
        self.scorer = self._get_scorer(scorer)
        self.max_matches = max_matches_per_entity
        self.n_workers = n_workers or min(mp.cpu_count() - 1, 4)
        # On Windows, default to ThreadPoolExecutor
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
    
    def _process_batch(self, args):
        """Process a batch of entities for matching"""
        batch_df, all_names, name_to_id = args
        results = []
        
        for _, row in batch_df.iterrows():
            entity_id = row['id']
            entity_name = row['name']
            
            if pd.isna(entity_name):
                continue
            
            # Find matches using RapidFuzz
            matches = process.extract(
                entity_name,
                all_names,
                scorer=self.scorer,
                limit=None
            )
            
            # Filter by threshold and exclude self-matches
            match_count = 0
            for match_name, score, _ in matches:
                if score >= self.threshold:
                    match_id = name_to_id.get(match_name)
                    if match_id and match_id != entity_id:
                        results.append({
                            'id1': entity_id,
                            'id2': match_id,
                            'name1': entity_name,
                            'name2': match_name,
                            'similarity_score': score
                        })
                        match_count += 1
                        
                        if match_count >= self.max_matches:
                            break
        
        return results
    
    def find_matches(self, df: pd.DataFrame, **kwargs) -> MatchResult:
        """Find potential duplicate pairs using fuzzy matching"""
        
        # Prepare data
        df_clean = df.dropna(subset=['name']).copy()
        
        if df_clean.empty:
            return MatchResult(
                pairs=pd.DataFrame(columns=['id1', 'id2', 'name1', 'name2', 'similarity_score']),
                metadata={'message': 'No valid entities to match'}
            )
        
        all_names = df_clean['name'].unique().tolist()
        name_to_id = dict(zip(df_clean['name'], df_clean['id']))
        
        # Split data into batches
        batch_size = max(1, len(df_clean) // (self.n_workers * 4))
        batches = []
        
        for i in range(0, len(df_clean), batch_size):
            batch_df = df_clean.iloc[i:i+batch_size]
            batches.append((batch_df, all_names, name_to_id))
        
        # Process batches (using ThreadPoolExecutor on Windows)
        all_results = []
        
        # Use ThreadPoolExecutor which works better on Windows
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
            # Remove duplicates (keeping highest score for each pair)
            pairs_df = pairs_df.sort_values('similarity_score', ascending=False)
            pairs_df = pairs_df.drop_duplicates(subset=['id1', 'id2'], keep='first')
        else:
            pairs_df = pd.DataFrame(columns=['id1', 'id2', 'name1', 'name2', 'similarity_score'])
        
        metadata = {
            'threshold': self.threshold,
            'scorer': self.scorer_name,
            'total_entities': len(df),
            'matches_found': len(pairs_df)
        }
        
        return MatchResult(pairs=pairs_df, metadata=metadata)