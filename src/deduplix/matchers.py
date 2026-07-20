# deduplix/matchers.py

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
    Fuzzy string matcher using rapidfuzz.process.cdist for vectorized
    pairwise similarity computation.

    rapidfuzz.process.cdist computes the full (n x m) similarity matrix
    between two lists of strings in C++ with optional parallelism, then
    numpy thresholding extracts candidate pairs — replacing the Python-level
    extract() loop entirely.

    Parameters
    ----------
    threshold : float, default=80.0
        Minimum similarity score (0-100) for a pair to be considered a match
    scorer : str, default='token_sort_ratio'
        RapidFuzz scoring algorithm:
        'ratio', 'partial_ratio', 'token_sort_ratio',
        'token_set_ratio', 'WRatio', 'jaro', 'jaro_winkler'
    max_matches_per_entity : int or None, default=None
        Cap matches per entity. None = unlimited.
    n_workers : int or None, default=None
        Workers passed to rapidfuzz.process.cdist. None = auto (all cores).
        -1 = all cores. 1 = single-threaded.
    batch_size : int, default=500
        Rows per batch when chunking large datasets to control peak memory.
        Full matrix for n=10000 at float32 = 10000x10000x4 bytes = 400MB.
        Batching keeps peak usage to batch_size x n x 4 bytes.
    lowercase : bool, default=True
        Lowercase text before matching
    strip_whitespace : bool, default=True
        Normalize whitespace before matching
    remove_punctuation : bool, default=False
        Remove punctuation before matching
    punctuation_pattern : str, default=r'[^\\w\\s]'
        Regex pattern for punctuation removal
    score_multiplier : float, default=1.0
        Multiplier applied to raw scores. Useful when scorer returns 0-1
        range (e.g., jaro, jaro_winkler) to rescale to 0-100.
        Set to 100.0 for jaro/jaro_winkler scorers.

    Examples
    --------
    Basic usage:
    >>> matcher = FuzzyMatcher(threshold=85.0, scorer='token_sort_ratio')
    >>> result = matcher.find_matches(df)

    High-performance with all cores:
    >>> matcher = FuzzyMatcher(
    ...     threshold=80.0,
    ...     scorer='WRatio',
    ...     n_workers=-1,       # all cores via rapidfuzz native parallelism
    ...     batch_size=1000     # 1000 x n_entities similarity matrix per batch
    ... )

    Cross-dataset:
    >>> result = matcher.find_cross_matches(
    ...     df1, df2,
    ...     id_column1='company_id',
    ...     name_column1='company_name',
    ...     id_column2='entity_id',
    ...     name_column2='name'
    ... )
    """

    def __init__(
        self,
        threshold: float = 80.0,
        scorer: str = 'token_sort_ratio',
        max_matches_per_entity: Optional[int] = None,
        n_workers: Optional[int] = 1,
        batch_size: int = 500,
        lowercase: bool = True,
        strip_whitespace: bool = True,
        remove_punctuation: bool = False,
        punctuation_pattern: str = r'[^\w\s]',
        score_multiplier: float = 1.0
    ):
        self.threshold = threshold
        self.scorer_name = scorer
        self.scorer = self._get_scorer(scorer)
        self.max_matches = max_matches_per_entity
        self.n_workers = n_workers  # None = let rapidfuzz decide
        self.batch_size = batch_size
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace
        self.remove_punctuation = remove_punctuation
        self.punctuation_pattern = punctuation_pattern
        self.score_multiplier = score_multiplier

    # ------------------------------------------------------------------
    # Scorer resolution
    # ------------------------------------------------------------------

    def _get_scorer(self, scorer_name: str):
        """
        Resolve scorer name to rapidfuzz callable.

        Parameters
        ----------
        scorer_name : str
            One of: ratio, partial_ratio, token_sort_ratio,
            token_set_ratio, WRatio, jaro, jaro_winkler

        Returns
        -------
        callable
            RapidFuzz scorer function
        """
        from rapidfuzz import distance as rfd

        scorers = {
            'ratio':            fuzz.ratio,
            'partial_ratio':    fuzz.partial_ratio,
            'token_sort_ratio': fuzz.token_sort_ratio,
            'token_set_ratio':  fuzz.token_set_ratio,
            'WRatio':           fuzz.WRatio,
            'jaro':             rfd.Jaro.normalized_similarity,
            'jaro_winkler':     rfd.JaroWinkler.normalized_similarity,
        }
        if scorer_name not in scorers:
            raise ValueError(
                f"Unknown scorer '{scorer_name}'. "
                f"Choose from: {list(scorers.keys())}"
            )
        return scorers[scorer_name]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single string before matching.

        Parameters
        ----------
        text : str
            Raw input string

        Returns
        -------
        str
            Preprocessed string, or '' if input is null/empty
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

    def _preprocess_series(self, names: pd.Series) -> List[str]:
        """
        Vectorized preprocessing of a name series.

        Parameters
        ----------
        names : pd.Series
            Series of raw name strings

        Returns
        -------
        List[str]
            Preprocessed strings (empty string for nulls)
        """
        return [self.preprocess_text(n) for n in names]

    # ------------------------------------------------------------------
    # Core cdist computation
    # ------------------------------------------------------------------

    def _cdist_batch(
        self,
        queries: List[str],
        choices: List[str],
        score_cutoff: float
    ) -> np.ndarray:
        """
        Compute similarity matrix for one batch using rapidfuzz.process.cdist.

        rapidfuzz.process.cdist:
        - Implemented in C++ with SIMD acceleration
        - Native multi-threading via n_workers parameter
        - Returns np.ndarray of shape (len(queries), len(choices))
        - dtype=np.float32 by default (saves memory vs float64)
        - score_cutoff sets values below threshold to 0 in-place,
          enabling sparse extraction without extra masking

        Parameters
        ----------
        queries : List[str]
            Preprocessed query strings (batch rows)
        choices : List[str]
            Preprocessed choice strings (all columns)
        score_cutoff : float
            Scores below this are set to 0 in the output matrix

        Returns
        -------
        np.ndarray
            Shape (len(queries), len(choices)), dtype float32
        """
        return process.cdist(
            queries,
            choices,
            scorer=self.scorer,
            score_cutoff=score_cutoff,
            workers=self.n_workers,
            dtype=np.float32
        )

    def _extract_pairs_from_matrix(
        self,
        score_matrix: np.ndarray,
        query_ids: List,
        query_names_original: List[str],
        choice_ids: List,
        choice_names_original: List[str],
        batch_offset: int,
        within_dataset: bool
    ) -> List[Dict]:
        """
        Extract matching pairs from a similarity score matrix.

        Uses np.nonzero (equivalent to np.where(matrix > 0)) which is
        faster than np.argwhere for sparse results, since score_cutoff
        already zeroed out non-matches in cdist.

        Parameters
        ----------
        score_matrix : np.ndarray
            Shape (batch_size, n_choices), values are similarity scores
            (0 = below threshold due to score_cutoff in cdist)
        query_ids : List
            Entity IDs for matrix rows
        query_names_original : List[str]
            Original (non-preprocessed) names for rows
        choice_ids : List
            Entity IDs for matrix columns
        choice_names_original : List[str]
            Original names for columns
        batch_offset : int
            Row index offset in the full matrix (for within-dataset
            symmetric pair elimination)
        within_dataset : bool
            If True, skip self-pairs and enforce canonical (i < j) ordering

        Returns
        -------
        List[Dict]
            Matched pairs: {id1, id2, name1, name2, similarity_score}
        """
        results = []

        # np.nonzero returns (row_indices, col_indices) of non-zero elements
        # This is fast because score_cutoff already zeroed out non-matches
        row_indices, col_indices = np.nonzero(score_matrix)

        for local_row, col in zip(row_indices, col_indices):
            global_row = batch_offset + local_row

            if within_dataset:
                # Skip self-pairs
                if global_row == col:
                    continue
                # Enforce canonical ordering: only keep (i, j) where i < j
                # This eliminates symmetric duplicates (A,B) and (B,A)
                if global_row >= col:
                    continue

            score = float(score_matrix[local_row, col]) * self.score_multiplier

            # Re-check after multiplier (in case multiplier > 1.0)
            if score < self.threshold:
                continue

            results.append({
                'id1': query_ids[local_row],
                'id2': choice_ids[col],
                'name1': query_names_original[local_row],
                'name2': choice_names_original[col],
                'similarity_score': score
            })

        return results

    def _apply_max_matches(
        self,
        pairs_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply max_matches_per_entity cap if configured.

        Keeps top-scoring matches per entity.

        Parameters
        ----------
        pairs_df : pd.DataFrame
            All matched pairs

        Returns
        -------
        pd.DataFrame
            Filtered pairs respecting max_matches cap
        """
        if self.max_matches is None or pairs_df.empty:
            return pairs_df

        # Count matches per entity (appearing as id1 or id2)
        # Sort descending so head() keeps best matches
        pairs_sorted = pairs_df.sort_values('similarity_score', ascending=False)

        # Track match counts per entity
        entity_counts: Dict[Any, int] = {}
        keep_indices = []

        for idx, row in pairs_sorted.iterrows():
            id1, id2 = row['id1'], row['id2']
            count1 = entity_counts.get(id1, 0)
            count2 = entity_counts.get(id2, 0)

            if count1 < self.max_matches and count2 < self.max_matches:
                keep_indices.append(idx)
                entity_counts[id1] = count1 + 1
                entity_counts[id2] = count2 + 1

        return pairs_df.loc[keep_indices]

    # ------------------------------------------------------------------
    # Within-dataset matching
    # ------------------------------------------------------------------

    def find_matches(self, df: pd.DataFrame, **kwargs) -> MatchResult:
        """
        Find duplicate pairs within a single dataset.

        Uses rapidfuzz.process.cdist in row-batches to compute the upper
        triangle of the full pairwise similarity matrix, avoiding both
        self-pairs and symmetric duplicates.

        Memory model
        ------------
        Each batch produces a (batch_size x n_entities) float32 matrix.
        Peak memory ≈ batch_size × n_entities × 4 bytes.
        Example: batch_size=500, n=10000 → 500×10000×4 = 20MB per batch.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'id' and 'name' columns
        **kwargs
            Unused (for interface compatibility)

        Returns
        -------
        MatchResult
            Matched pairs sorted by descending similarity score
        """
        df_clean = df.dropna(subset=['name']).copy()

        if df_clean.empty:
            return MatchResult(
                pairs=pd.DataFrame(
                    columns=['id1', 'id2', 'name1', 'name2', 'similarity_score']
                ),
                metadata={'message': 'No valid entities to match'}
            )

        ids = df_clean['id'].tolist()
        names_original = df_clean['name'].tolist()
        names_preprocessed = self._preprocess_series(df_clean['name'])

        # Filter entities with empty preprocessed names
        valid = [
            (id_, orig, prep)
            for id_, orig, prep in zip(ids, names_original, names_preprocessed)
            if prep
        ]
        if len(valid) < 2:
            return MatchResult(
                pairs=pd.DataFrame(
                    columns=['id1', 'id2', 'name1', 'name2', 'similarity_score']
                ),
                metadata={'message': 'Insufficient valid entities after preprocessing'}
            )

        ids, names_original, names_preprocessed = map(list, zip(*valid))
        n = len(ids)

        print(f"  Computing pairwise similarity for {n:,} entities "
              f"using rapidfuzz.process.cdist...")

        all_results = []

        # Process in row-batches to control memory
        # For within-dataset: batch rows i, compare against all j > i
        # to naturally get the upper triangle (no symmetric duplicates)
        for batch_start in tqdm(
            range(0, n, self.batch_size),
            desc="cdist batches",
            unit="batch"
        ):
            batch_end = min(batch_start + self.batch_size, n)

            batch_queries_prep = names_preprocessed[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]
            batch_names_orig = names_original[batch_start:batch_end]

            # Only compare against entities with index > batch_start
            # to get upper triangle and avoid symmetric duplicates
            choices_prep = names_preprocessed[batch_start:]
            choices_ids = ids[batch_start:]
            choices_names_orig = names_original[batch_start:]

            if not choices_prep:
                continue

            # Compute similarity matrix for this batch
            # Shape: (batch_size, n - batch_start)
            score_matrix = self._cdist_batch(
                batch_queries_prep,
                choices_prep,
                score_cutoff=self.threshold
            )

            # Extract pairs, using local offset within the choices slice
            # Within-dataset: skip diagonal (self-pairs) and lower triangle
            batch_results = self._extract_pairs_from_matrix(
                score_matrix=score_matrix,
                query_ids=batch_ids,
                query_names_original=batch_names_orig,
                choice_ids=choices_ids,
                choice_names_original=choices_names_orig,
                batch_offset=0,          # local offset: row 0 = entity batch_start
                within_dataset=True
            )
            all_results.extend(batch_results)

        if all_results:
            pairs_df = pd.DataFrame(all_results)
            pairs_df = pairs_df.sort_values(
                'similarity_score', ascending=False
            ).reset_index(drop=True)

            if self.max_matches is not None:
                pairs_df = self._apply_max_matches(pairs_df)
        else:
            pairs_df = pd.DataFrame(
                columns=['id1', 'id2', 'name1', 'name2', 'similarity_score']
            )

        metadata = {
            'threshold': self.threshold,
            'scorer': self.scorer_name,
            'total_entities': n,
            'matches_found': len(pairs_df),
            'batches_processed': (n + self.batch_size - 1) // self.batch_size,
            'preprocessing': {
                'lowercase': self.lowercase,
                'strip_whitespace': self.strip_whitespace,
                'remove_punctuation': self.remove_punctuation
            }
        }

        return MatchResult(pairs=pairs_df, metadata=metadata)

    # ------------------------------------------------------------------
    # Cross-dataset matching
    # ------------------------------------------------------------------

    def _find_cross_matches_single_column(
        self,
        df1_ids: List,
        df1_names_orig: List[str],
        df1_names_prep: List[str],
        df2_ids: List,
        df2_names_orig: List[str],
        df2_names_prep: List[str],
        col1: str = 'name',
        col2: str = 'name'
    ) -> List[Dict]:
        """
        Compute cross-dataset matches for one column pair using cdist.

        No upper-triangle optimization needed: every (df1_i, df2_j) pair
        is a valid cross-dataset match, so we process the full matrix.

        Parameters
        ----------
        df1_ids, df2_ids : List
            Entity IDs from each dataset
        df1_names_orig, df2_names_orig : List[str]
            Original names for output
        df1_names_prep, df2_names_prep : List[str]
            Preprocessed names for scoring
        col1, col2 : str
            Column names (for metadata in multi-column mode)

        Returns
        -------
        List[Dict]
            Matched pairs with similarity scores
        """
        n1 = len(df1_ids)
        all_results = []

        for batch_start in range(0, n1, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n1)

            batch_queries_prep = df1_names_prep[batch_start:batch_end]
            batch_ids = df1_ids[batch_start:batch_end]
            batch_names_orig = df1_names_orig[batch_start:batch_end]

            # Full cross-product: compare batch against ALL df2 entities
            score_matrix = self._cdist_batch(
                batch_queries_prep,
                df2_names_prep,
                score_cutoff=self.threshold
            )

            # Extract pairs — no within_dataset restrictions
            batch_results = self._extract_pairs_from_matrix(
                score_matrix=score_matrix,
                query_ids=batch_ids,
                query_names_original=batch_names_orig,
                choice_ids=df2_ids,
                choice_names_original=df2_names_orig,
                batch_offset=batch_start,
                within_dataset=False
            )

            # Tag with matched column names for multi-column mode
            for r in batch_results:
                r['matched_column1'] = col1
                r['matched_column2'] = col2
                r['source1'] = 'df1'
                r['source2'] = 'df2'

            all_results.extend(batch_results)

        return all_results

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
        Cross-dataset matching across multiple column combinations.

        For each (col1, col2) combination, runs cdist and keeps the
        best-scoring match per entity pair across all combinations.

        Parameters
        ----------
        df1, df2 : pd.DataFrame
            Input datasets
        id_column1, id_column2 : str
            ID columns
        name_columns1, name_columns2 : List[str]
            Name columns to try from each dataset

        Returns
        -------
        MatchResult
            Best matches across all column combinations
        """
        total_combinations = len(name_columns1) * len(name_columns2)
        print(f"  Trying {total_combinations} column combination(s) "
              f"with rapidfuzz.process.cdist...")

        # Pre-process and cache each column to avoid redundant work
        df1_cache: Dict[str, Tuple[List, List[str], List[str]]] = {}
        for col in name_columns1:
            if col not in df1.columns:
                print(f"  Warning: column '{col}' not in df1, skipping")
                continue
            subset = df1[[id_column1, col]].dropna(subset=[col]).copy()
            if subset.empty:
                continue
            ids_ = subset[id_column1].tolist()
            orig_ = subset[col].tolist()
            prep_ = self._preprocess_series(subset[col])
            # Filter empty preprocessed
            valid = [(i, o, p) for i, o, p in zip(ids_, orig_, prep_) if p]
            if valid:
                df1_cache[col] = tuple(map(list, zip(*valid)))

        df2_cache: Dict[str, Tuple[List, List[str], List[str]]] = {}
        for col in name_columns2:
            if col not in df2.columns:
                print(f"  Warning: column '{col}' not in df2, skipping")
                continue
            subset = df2[[id_column2, col]].dropna(subset=[col]).copy()
            if subset.empty:
                continue
            ids_ = subset[id_column2].tolist()
            orig_ = subset[col].tolist()
            prep_ = self._preprocess_series(subset[col])
            valid = [(i, o, p) for i, o, p in zip(ids_, orig_, prep_) if p]
            if valid:
                df2_cache[col] = tuple(map(list, zip(*valid)))

        all_matches: List[Dict] = []

        for col1 in name_columns1:
            if col1 not in df1_cache:
                continue
            df1_ids, df1_orig, df1_prep = df1_cache[col1]

            for col2 in name_columns2:
                if col2 not in df2_cache:
                    continue
                df2_ids, df2_orig, df2_prep = df2_cache[col2]

                print(f"    {col1} × {col2}: "
                      f"{len(df1_ids)} × {len(df2_ids)} entities")

                matches = self._find_cross_matches_single_column(
                    df1_ids, df1_orig, df1_prep,
                    df2_ids, df2_orig, df2_prep,
                    col1=col1, col2=col2
                )
                all_matches.extend(matches)

        if not all_matches:
            return MatchResult(
                pairs=pd.DataFrame(columns=[
                    'id1', 'id2', 'name1', 'name2', 'similarity_score',
                    'source1', 'source2', 'matched_column1', 'matched_column2'
                ]),
                metadata={'message': 'No matches found across any column combination'}
            )

        matches_df = pd.DataFrame(all_matches)

        # Deduplicate: keep best-scoring match per (df1_id, df2_id) pair
        matches_df = (
            matches_df
            .sort_values('similarity_score', ascending=False)
            .drop_duplicates(subset=['id1', 'id2'])
            .reset_index(drop=True)
        )

        print(f"  Total matches: {len(all_matches):,} → "
              f"{len(matches_df):,} unique pairs after deduplication")

        metadata = {
            'threshold': self.threshold,
            'scorer': self.scorer_name,
            'df1_entities': len(df1),
            'df2_entities': len(df2),
            'cross_matches_found': len(matches_df),
            'matching_type': 'cross_dataset_multi_column',
            'name_columns1': name_columns1,
            'name_columns2': name_columns2,
            'total_combinations_tried': total_combinations,
            'preprocessing': {
                'lowercase': self.lowercase,
                'strip_whitespace': self.strip_whitespace,
                'remove_punctuation': self.remove_punctuation
            }
        }

        return MatchResult(pairs=matches_df, metadata=metadata)

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
        Find matches between two datasets using rapidfuzz.process.cdist.

        Parameters
        ----------
        df1, df2 : pd.DataFrame
            Datasets to match
        id_column1, id_column2 : str
            ID column names
        name_column1, name_column2 : str
            Primary name column names
        name_columns1, name_columns2 : List[str], optional
            Multiple name columns to try. If provided, all combinations
            are tried and best match per pair is kept.
        **kwargs
            Unused

        Returns
        -------
        MatchResult
            Cross-dataset matches with similarity scores
        """
        # Route to multi-column handler if needed
        if name_columns1 or name_columns2:
            cols1 = name_columns1 or [name_column1]
            cols2 = name_columns2 or [name_column2]
            return self._find_cross_matches_multi_column(
                df1, df2, id_column1, id_column2, cols1, cols2
            )

        # Single-column path
        df1_clean = df1[[id_column1, name_column1]].dropna(
            subset=[name_column1]
        ).copy()
        df2_clean = df2[[id_column2, name_column2]].dropna(
            subset=[name_column2]
        ).copy()

        if df1_clean.empty or df2_clean.empty:
            return MatchResult(
                pairs=pd.DataFrame(
                    columns=['id1', 'id2', 'name1', 'name2',
                             'similarity_score', 'source1', 'source2']
                ),
                metadata={'message': 'One or both datasets are empty'}
            )

        df1_ids = df1_clean[id_column1].tolist()
        df1_orig = df1_clean[name_column1].tolist()
        df1_prep = self._preprocess_series(df1_clean[name_column1])

        df2_ids = df2_clean[id_column2].tolist()
        df2_orig = df2_clean[name_column2].tolist()
        df2_prep = self._preprocess_series(df2_clean[name_column2])

        # Filter empty preprocessed names
        valid1 = [(i, o, p) for i, o, p in zip(df1_ids, df1_orig, df1_prep) if p]
        valid2 = [(i, o, p) for i, o, p in zip(df2_ids, df2_orig, df2_prep) if p]

        if not valid1 or not valid2:
            return MatchResult(
                pairs=pd.DataFrame(
                    columns=['id1', 'id2', 'name1', 'name2',
                             'similarity_score', 'source1', 'source2']
                ),
                metadata={'message': 'No valid entities after preprocessing'}
            )

        df1_ids, df1_orig, df1_prep = map(list, zip(*valid1))
        df2_ids, df2_orig, df2_prep = map(list, zip(*valid2))

        print(f"  Cross-dataset cdist: {len(df1_ids):,} × {len(df2_ids):,} "
              f"= {len(df1_ids) * len(df2_ids):,} pairs to evaluate")

        all_results = self._find_cross_matches_single_column(
            df1_ids, df1_orig, df1_prep,
            df2_ids, df2_orig, df2_prep
        )

        if all_results:
            pairs_df = pd.DataFrame(all_results)
            pairs_df = pairs_df.sort_values(
                'similarity_score', ascending=False
            ).reset_index(drop=True)

            if self.max_matches is not None:
                pairs_df = self._apply_max_matches(pairs_df)
        else:
            pairs_df = pd.DataFrame(
                columns=['id1', 'id2', 'name1', 'name2',
                         'similarity_score', 'source1', 'source2']
            )

        metadata = {
            'threshold': self.threshold,
            'scorer': self.scorer_name,
            'df1_entities': len(df1_ids),
            'df2_entities': len(df2_ids),
            'cross_matches_found': len(pairs_df),
            'matching_type': 'cross_dataset',
            'preprocessing': {
                'lowercase': self.lowercase,
                'strip_whitespace': self.strip_whitespace,
                'remove_punctuation': self.remove_punctuation
            }
        }

        return MatchResult(pairs=pairs_df, metadata=metadata)