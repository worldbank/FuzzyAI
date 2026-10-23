"""Pair validators: decide whether candidate duplicate *pairs* are real matches.

(Input-data checks live in :mod:`deduplix.validation`.)
"""
import os
import re
import json
import time
import warnings
from typing import Optional, List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from .core import Validator, ValidationResult, MatchResult
from .exceptions import DeduplixWarning


class RuleBasedValidator(Validator):
    """Simple Rule-based validation """

    def __init__(
        self,
        min_score: float = 90.0,
        metadata_rules: Optional[List[Dict[str, Any]]] = None,
        custom_rules: Optional[List[Callable]] = None
    ):
        """
        Parameters
        ----------
        min_score : float
            Minimum similarity score threshold (always applied)
        metadata_rules : Optional[List[Dict[str, Any]]]
            List of metadata rules. Each rule is a dict with:
            {
                'column': 'country',           # Column name to check
                'operation': 'exact',          # 'exact', 'fuzzy', 'inequality'
                'fuzzy_threshold': 80,         # For fuzzy matching (optional)
                'max_diff_percent': 50,        # For inequality (optional)
                'comparator': '<='             # For inequality: '<', '>', '<=', '>=' (optional)
            }
        custom_rules : Optional[List[Callable]]
            Custom rule functions: func(row, original_df) -> bool

        Examples
        --------
        >>> validator = RuleBasedValidator(
        ...     min_score=85.0,
        ...     metadata_rules=[
        ...         {'column': 'country', 'operation': 'exact'},
        ...         {'column': 'industry', 'operation': 'fuzzy', 'fuzzy_threshold': 80},
        ...         {'column': 'revenue', 'operation': 'inequality', 'max_diff_percent': 50}
        ...     ]
        ... )
        """
        self.min_score = min_score
        self.metadata_rules = metadata_rules or []
        self.custom_rules = custom_rules or []

    def _check_score(self, row: pd.Series) -> tuple:
        """Check similarity score threshold"""
        if row['similarity_score'] >= self.min_score:
            return True, ""
        return False, f"Score {row['similarity_score']:.1f} < threshold {self.min_score}"

    def _index(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """id -> first row, built once per original frame (replaces a filter per pair)."""
        cached = getattr(self, '_index_cache', None)
        if cached is None or cached[0] is not original_df:
            idx = original_df.drop_duplicates(subset='id', keep='first').set_index('id')
            self._index_cache = (original_df, idx)
        return self._index_cache[1]

    def _pair_values(self, row, original_df: pd.DataFrame, column: str):
        """Values of ``column`` for both sides of the pair, or None if either id is unknown."""
        if column not in original_df.columns:
            return None
        idx = self._index(original_df)
        if row['id1'] not in idx.index or row['id2'] not in idx.index:
            return None
        return idx.at[row['id1'], column], idx.at[row['id2'], column]

    def _check_exact_match(self, row, original_df: pd.DataFrame, column: str) -> tuple:
        """Check if column values match exactly"""
        vals = self._pair_values(row, original_df, column)
        if vals is None:
            return True, ""
        val1, val2 = vals
        if pd.notna(val1) and pd.notna(val2) and val1 != val2:
            return False, f"{column}: {val1} != {val2}"
        return True, ""

    def _check_fuzzy_match(self, row, original_df: pd.DataFrame, column: str, threshold: float = 80.0) -> tuple:
        """Check if column values fuzzy match"""
        vals = self._pair_values(row, original_df, column)
        if vals is None:
            return True, ""
        from rapidfuzz import fuzz

        val1, val2 = vals
        if pd.notna(val1) and pd.notna(val2):
            score = fuzz.ratio(str(val1), str(val2))
            if score < threshold:
                return False, f"{column}: fuzzy score {score:.1f} < {threshold}"
        return True, ""

    def _check_inequality(self, row, original_df: pd.DataFrame, column: str,
                          max_diff_percent: float = None, comparator: str = '<=') -> tuple:
        """Check if numeric column values are within threshold"""
        vals = self._pair_values(row, original_df, column)
        if vals is None:
            return True, ""
        val1, val2 = vals
        if pd.notna(val1) and pd.notna(val2):
            try:
                val1 = float(val1)
                val2 = float(val2)

                if max_diff_percent is not None:
                    max_val = max(val1, val2)
                    if max_val > 0:
                        diff_percent = abs(val1 - val2) / max_val * 100
                        if diff_percent > max_diff_percent:
                            return False, f"{column}: diff {diff_percent:.1f}% > {max_diff_percent}%"
                else:
                    diff = abs(val1 - val2)
                    if comparator == '<' and not (diff < max_diff_percent):
                        return False, f"{column}: diff {diff} not < {max_diff_percent}"
                    elif comparator == '>' and not (diff > max_diff_percent):
                        return False, f"{column}: diff {diff} not > {max_diff_percent}"
                    elif comparator == '<=' and not (diff <= max_diff_percent):
                        return False, f"{column}: diff {diff} not <= {max_diff_percent}"
                    elif comparator == '>=' and not (diff >= max_diff_percent):
                        return False, f"{column}: diff {diff} not >= {max_diff_percent}"

            except (ValueError, TypeError):
                pass

        return True, ""

    def validate(self, match_result: MatchResult, original_df: pd.DataFrame = None, **kwargs) -> ValidationResult:
        """Apply rules to validate matches"""

        if match_result.pairs.empty:
            return ValidationResult(
                validated_pairs=pd.DataFrame(),
                removed_pairs=pd.DataFrame(),
                metadata={'message': 'No pairs to validate'}
            )

        validated = []
        removed = []

        for row in match_result.pairs.to_dict('records'):
            passed = True
            reasons = []

            score_passed, score_reason = self._check_score(row)
            if not score_passed:
                passed = False
                reasons.append(score_reason)

            if original_df is not None:
                for rule in self.metadata_rules:
                    column = rule.get('column')
                    operation = rule.get('operation', 'exact')

                    if operation == 'exact':
                        rule_passed, reason = self._check_exact_match(row, original_df, column)
                    elif operation == 'fuzzy':
                        threshold = rule.get('fuzzy_threshold', 80.0)
                        rule_passed, reason = self._check_fuzzy_match(row, original_df, column, threshold)
                    elif operation == 'inequality':
                        max_diff = rule.get('max_diff_percent')
                        comparator = rule.get('comparator', '<=')
                        rule_passed, reason = self._check_inequality(row, original_df, column, max_diff, comparator)
                    else:
                        continue

                    if not rule_passed:
                        passed = False
                        reasons.append(reason)

            for custom_rule in self.custom_rules:
                try:
                    if not custom_rule(pd.Series(row), original_df):
                        passed = False
                        reasons.append(f"Custom rule failed: {custom_rule.__name__}")
                except Exception as e:
                    print(f"Error in custom rule {custom_rule.__name__}: {e}")

            row_dict = dict(row)
            row_dict['validation_reason'] = "; ".join(reasons) if reasons else "Passed all rules"

            if passed:
                validated.append(row_dict)
            else:
                removed.append(row_dict)

        validated_df = pd.DataFrame(validated) if validated else pd.DataFrame()
        removed_df = pd.DataFrame(removed) if removed else pd.DataFrame()

        metadata = {
            'min_score': self.min_score,
            'metadata_rules_count': len(self.metadata_rules),
            'custom_rules_count': len(self.custom_rules),
            'validated_count': len(validated_df),
            'removed_count': len(removed_df)
        }

        return ValidationResult(
            validated_pairs=validated_df,
            removed_pairs=removed_df,
            metadata=metadata
        )

class LLMValidator(Validator):
    """
    Enhanced LLM-based validation of matches with support for:
      - provider="openai"      => OpenAI >= 1.x (OpenAI client)
      - provider="anthropic"   => Anthropic v1 Messages API
      - provider="databricks"  => Databricks Foundation Models API via OpenAI-compatible endpoint,
                                  or LangChain ChatDatabricks if use_langchain_databricks=True

    Features:
      - Custom matching rules and metadata-aware validation
      - Multi-provider LLM support (OpenAI, Anthropic, Databricks)
      - Batch processing with parallel execution
      - Pre-filtering based on hard rules
      - Robust error handling and retries

    Parameters
    ----------
    api_key : Optional[str]
        API key for the selected provider. If None, reads standard env vars:
          - OPENAI_API_KEY for provider="openai"
          - ANTHROPIC_API_KEY for provider="anthropic"
          - DATABRICKS_TOKEN or DATABRICKS_API_KEY for provider="databricks"
    model : str
        Model name (for Databricks, this is the *serving endpoint name* if not overridden).
    provider : str
        "openai" | "anthropic" | "databricks"
    batch_size : int
        Number of pairs per LLM call.
    n_workers : int
        Parallel batches (thread pool).
    temperature : float
        Sampling temperature.
    max_retries : int
        Retry attempts per batch.
    custom_rules : Optional[Dict[str, Any]]
        Custom rules for matching. Examples:
        {
            "custom_instructions": "Additional instructions for the LLM"
        }
    metadata_columns : Optional[List[str]]
        Columns to include in LLM context (e.g., ['country', 'industry'])
    databricks_host : Optional[str]
        Your workspace base URL, e.g. "https://adb-1234567890123456.7.azuredatabricks.net".
        If None, reads DATABRICKS_HOST.
    databricks_endpoint : Optional[str]
        Serving endpoint name. If None, uses `model`.
    use_langchain_databricks : bool
        If True, try LangChain's ChatDatabricks instead of OpenAI-compatible client.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        batch_size: int = 10,
        n_workers: int = 4,
        temperature: float = 0.0,
        max_retries: int = 3,
        custom_rules: Optional[Dict[str, Any]] = None,
        metadata_columns: Optional[List[str]] = None,
        checkpoint_every_n_batches: int = 0,

        databricks_host: Optional[str] = None,
        databricks_endpoint: Optional[str] = None,
        use_langchain_databricks: bool = False,

        on_failure: str = "match",
        prompt_template: str = "pair_validation_v1.j2",
        parser=None,
        max_tokens: int = 4096,
        cache=None,
    ):
        """Initialize LLM Validator"""
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.provider = provider.lower()
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.temperature = temperature
        self.max_retries = max_retries
        self.custom_rules = custom_rules or {}
        self.metadata_columns = metadata_columns or []
        self.checkpoint_every_n_batches = checkpoint_every_n_batches

        self.databricks_host = databricks_host or os.getenv("DATABRICKS_HOST")
        self.databricks_endpoint = databricks_endpoint
        self.use_langchain_databricks = use_langchain_databricks

        if on_failure not in ("match", "non_match", "unvalidated"):
            raise ValueError("on_failure must be 'match', 'non_match' or 'unvalidated'")
        self.on_failure = on_failure
        self.prompt_template = prompt_template
        self.parser = parser
        self.max_tokens = max_tokens
        self.cache = cache
        self.failed_batches = 0
        self.failed_pairs = 0

        self.client = None
        self._client_mode = None
        self._init_client()

    # ---------- Client initialization ----------

    def _init_client(self):
        """Initialize the appropriate LLM client based on provider"""
        if self.provider == "openai":
            self._init_openai_v1()
        elif self.provider == "anthropic":
            self._init_anthropic_v1()
        elif self.provider == "databricks":
            self._init_databricks()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _init_openai_v1(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
        except Exception as e:
            raise ImportError("Install openai>=1.0: pip install openai") from e

        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY missing.")
        self.client = OpenAI(api_key=key)
        self._client_mode = "openai_v1"

    def _init_anthropic_v1(self):
        """Initialize Anthropic client"""
        try:
            from anthropic import Anthropic
        except Exception as e:
            raise ImportError("Install anthropic>=0.30: pip install anthropic") from e

        key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY missing.")
        self.client = Anthropic(api_key=key)
        self._client_mode = "anthropic_v1"

    def _init_databricks(self):
        """Initialize Databricks client (OpenAI-compatible or LangChain)"""
        if not self.use_langchain_databricks:
            try:
                from openai import OpenAI
            except Exception as e:
                raise ImportError(
                    "Databricks (OpenAI-compatible) requires openai>=1.0. Install: pip install openai"
                ) from e

            token = (
                self.api_key
                or os.getenv("DATABRICKS_TOKEN")
                or os.getenv("DATABRICKS_API_KEY")
            )
            if not token:
                raise ValueError("Databricks token missing (DATABRICKS_TOKEN or DATABRICKS_API_KEY).")
            if not self.databricks_host:
                raise ValueError("Databricks host missing (databricks_host or DATABRICKS_HOST).")

            base_url = self.databricks_host.rstrip("/") + "/serving-endpoints"
            self.client = OpenAI(api_key=token, base_url=base_url)
            self._client_mode = "databricks_openai"
            return

        try:
            from langchain_community.chat_models import ChatDatabricks
        except Exception as e:
            raise ImportError(
                "LangChain ChatDatabricks not installed. pip install langchain-community"
            ) from e

        token = (
            self.api_key
            or os.getenv("DATABRICKS_TOKEN")
            or os.getenv("DATABRICKS_API_KEY")
        )
        if not token:
            raise ValueError("Databricks token missing (DATABRICKS_TOKEN or DATABRICKS_API_KEY).")
        if not self.databricks_host:
            raise ValueError("Databricks host missing (databricks_host or DATABRICKS_HOST).")

        endpoint = self.databricks_endpoint or self.model
        self.client = ChatDatabricks(
            server_url=self.databricks_host.rstrip("/"),
            endpoint=endpoint,
            api_key=token,
            temperature=self.temperature,
        )
        self._client_mode = "databricks_langchain"

    # ---------- Prompt creation ----------

    def _metadata_index(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """id -> first row, built once per original frame (not one filter per pair)."""
        cached = getattr(self, '_meta_cache', None)
        if cached is None or cached[0] is not original_df:
            self._meta_cache = (original_df, original_df.drop_duplicates(subset='id', keep='first').set_index('id'))
        return self._meta_cache[1]

    def _engine_mode(self) -> bool:
        return (getattr(self, "parser", None) is not None
                or getattr(self, "prompt_template", "pair_validation_v1.j2") != "pair_validation_v1.j2")

    def _create_prompt(self, batch: pd.DataFrame, original_df: pd.DataFrame = None) -> str:
        """Create prompt for LLM validation with optional metadata support"""
        if self._engine_mode():
            return self._create_engine_prompt(batch, original_df)

        pairs_text = []
        for idx, row in batch.iterrows():
            pair_desc = f"Pair {idx}: '{row['name1']}' vs '{row['name2']}' (similarity: {row['similarity_score']:.1f}%)"

            if original_df is not None and self.metadata_columns:
                metadata_parts = []
                lookup = self._metadata_index(original_df)
                for col in self.metadata_columns:
                    if col in original_df.columns:
                        val1 = lookup.at[row['id1'], col] if row['id1'] in lookup.index else 'N/A'
                        val2 = lookup.at[row['id2'], col] if row['id2'] in lookup.index else 'N/A'
                        metadata_parts.append(f"{col}: {val1} vs {val2}")

                if metadata_parts:
                    pair_desc += f"\n  Metadata: {', '.join(metadata_parts)}"

            pairs_text.append(pair_desc)

        base_rules = [
            "- Carefully determine whether each pair are duplicated.",
        ]

        if 'custom_instructions' in self.custom_rules:
            base_rules.append(f"- {self.custom_rules['custom_instructions']}")

        prompt = (
            "Analyze these potential duplicate entity pairs. Return ONLY JSON with a top-level key 'decisions', "
            "where each item is {\"pair_index\": <index>, \"is_duplicate\": true|false, \"reason\": \"...\"}.\n\n"
            "Consider these rules:\n"
            f"{chr(10).join(base_rules)}\n\n"
            "Entity Pairs:\n"
            f"{chr(10).join(pairs_text)}\n\n"
            "Return only JSON."
        )

        return prompt

    def _create_engine_prompt(self, batch: pd.DataFrame, original_df: pd.DataFrame = None) -> str:
        """Render a template from deduplix.llm; pair_index is the local position 0..k-1."""
        from .llm.templates import PromptTemplate

        lookup = self._metadata_index(original_df) if original_df is not None else None
        pairs = []
        for pos, (_, row) in enumerate(batch.iterrows()):
            left, right = {"name": row['name1']}, {"name": row['name2']}
            for col in self.metadata_columns:
                if lookup is not None and col in lookup.columns:
                    left[col] = lookup.at[row['id1'], col] if row['id1'] in lookup.index else None
                    right[col] = lookup.at[row['id2'], col] if row['id2'] in lookup.index else None
            pairs.append({"idx": pos, "left": left, "right": right,
                          "features": {"similarity_score": row['similarity_score']}})
        instructions = self.custom_rules.get('custom_instructions')
        return PromptTemplate.load(self.prompt_template).render(
            pairs, instructions=instructions, metadata_columns=self.metadata_columns)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with multiple fallback strategies"""
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            return json.loads(fence.group(1))

        obj = re.search(r"\{.*\}", text, re.DOTALL)
        if obj:
            return json.loads(obj.group(0))

        return json.loads(text)

    # ---------- LLM API calls ----------

    def _call_llm(self, prompt: str) -> str:
        """Make LLM API call based on configured provider"""
        if self._client_mode == "openai_v1":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Identify duplicate entities. Reply with strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            return resp.choices[0].message.content

        if self._client_mode == "databricks_openai":
            model = self.databricks_endpoint or self.model
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Identify duplicate entities. Reply with strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            return resp.choices[0].message.content

        if self._client_mode == "databricks_langchain":
            result = self.client.invoke(prompt)
            return getattr(result, "content", str(result))

        if self._client_mode == "anthropic_v1":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="Identify duplicate entities. Reply with strict JSON only.",
                messages=[{"role": "user", "content": prompt}],
            )
            parts = []
            for block in getattr(resp, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts).strip()

        raise RuntimeError("LLM client not initialized correctly.")

    # ---------- Batch processing ----------

    def _process_batch(self, batch: pd.DataFrame, original_df: pd.DataFrame = None) -> Dict[int, Dict[str, Any]]:
        """Process a batch of pairs with retries and error handling"""
        prompt = self._create_prompt(batch, original_df)

        for attempt in range(self.max_retries):
            try:
                out = self._call_llm(prompt)
                decisions: Dict[int, Dict[str, Any]] = {}
                if self.parser is not None:
                    from .engine.registry import build
                    from .llm.providers.base import ProviderResponse

                    parse = build("parser", self.parser)
                    index = list(batch.index)
                    for d in parse(ProviderResponse(text=out), index):
                        decisions[index[d.pair_index]] = {"is_duplicate": bool(d.is_duplicate),
                                                          "reason": d.reason or ""}
                    return decisions
                data = self._extract_json(out)
                for d in data.get("decisions", []):
                    idx = d.get("pair_index")
                    if self._engine_mode() and isinstance(idx, int) and 0 <= idx < len(batch):
                        idx = batch.index[idx]  # templates number pairs locally
                    if idx in batch.index:
                        decisions[idx] = {
                            "is_duplicate": bool(d.get("is_duplicate", True)),
                            "reason": d.get("reason", ""),
                        }
                return decisions
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"Batch processing failed after {self.max_retries} attempts: {e}")
                    self.failed_batches += 1
                    self.failed_pairs += len(batch)
                    keep = self.on_failure != "non_match"
                    reason = ("LLM validation failed" if self.on_failure == "match"
                              else f"LLM validation failed ({self.on_failure})")
                    return {idx: {"is_duplicate": keep, "reason": reason} for idx in batch.index}

    # ---------- Public API ----------

    def validate(self, match_result: MatchResult, original_df: pd.DataFrame = None, **kwargs) -> ValidationResult:
        """
        Validate matches with optional metadata context and custom rules

        Parameters
        ----------
        match_result : MatchResult
            The matches to validate
        original_df : pd.DataFrame, optional
            Original dataframe with metadata columns for enhanced validation
        **kwargs
            Additional parameters:
            - checkpointer: Checkpointer instance for saving progress
            - data_hash: Hash string for checkpoint files
            - resume: Whether to resume from checkpoint

        Returns
        -------
        ValidationResult
            Results of the validation process
        """
        pairs_df = match_result.pairs.copy()
        self.failed_batches = 0
        self.failed_pairs = 0

        if pairs_df.empty:
            return ValidationResult(
                validated_pairs=pd.DataFrame(),
                removed_pairs=pd.DataFrame(),
                metadata={"message": "No pairs to validate"},
            )

        checkpointer = kwargs.get('checkpointer')
        data_hash = kwargs.get('data_hash')
        resume = kwargs.get('resume', True)

        batches = [
            pairs_df.iloc[i : i + self.batch_size]
            for i in range(0, len(pairs_df), self.batch_size)
        ]

        all_decisions: Dict[int, Dict[str, Any]] = {}
        processed_batches = set()

        if self.checkpoint_every_n_batches > 0 and checkpointer and data_hash and resume:
            checkpoint_data = checkpointer.load('validation_progress', data_hash)
            if checkpoint_data is not None:
                for _, row in checkpoint_data.iterrows():
                    idx = row['pair_index']
                    all_decisions[idx] = {
                        'is_duplicate': row['is_duplicate'],
                        'reason': row.get('reason', '')
                    }

                processed_indices = set(all_decisions.keys())
                for batch_idx, batch in enumerate(batches):
                    if all(idx in processed_indices for idx in batch.index):
                        processed_batches.add(batch_idx)

                print(f"  Resuming from checkpoint: {len(processed_batches)}/{len(batches)} batches already processed")

        batches_to_process = [(i, batch) for i, batch in enumerate(batches) if i not in processed_batches]

        if batches_to_process:
            batch_counter = len(processed_batches)

            with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
                futures = {
                    ex.submit(self._process_batch, batch, original_df): (batch_idx, batch)
                    for batch_idx, batch in batches_to_process
                }

                for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM validation"):
                    try:
                        batch_decisions = fut.result()
                        all_decisions.update(batch_decisions)
                        batch_counter += 1

                        if (self.checkpoint_every_n_batches > 0 and
                            checkpointer and
                            data_hash and
                            batch_counter % self.checkpoint_every_n_batches == 0):

                            progress_rows = []
                            for idx, decision in all_decisions.items():
                                progress_rows.append({
                                    'pair_index': idx,
                                    'is_duplicate': decision['is_duplicate'],
                                    'reason': decision['reason']
                                })
                            progress_df = pd.DataFrame(progress_rows)
                            checkpointer.save(progress_df, 'validation_progress', data_hash)
                            print(f"  Checkpoint saved: {batch_counter}/{len(batches)} batches processed")

                    except Exception as e:
                        print(f"Error processing batch: {e}")

        validated_rows, removed_rows = [], []

        for idx, row in pairs_df.iterrows():
            row_dict = row.to_dict()
            if idx in all_decisions:
                dec = all_decisions[idx]
                row_dict["validation_reason"] = dec.get("reason", "")
                if dec.get("is_duplicate", True):
                    validated_rows.append(row_dict)
                else:
                    removed_rows.append(row_dict)
            else:
                row_dict["validation_reason"] = "No LLM decision - kept by default"
                validated_rows.append(row_dict)

        validated_df = pd.DataFrame(validated_rows) if validated_rows else pd.DataFrame()
        removed_df = pd.DataFrame(removed_rows) if removed_rows else pd.DataFrame()

        meta = {
            "model": self.model if self.provider != "databricks" else (self.databricks_endpoint or self.model),
            "provider": self.provider,
            "batches_processed": len(batches),
            "batches_from_checkpoint": len(processed_batches),
            "validated_count": len(validated_df),
            "removed_count": len(removed_df),
            "validation_rate": len(validated_df) / len(match_result.pairs) if len(match_result.pairs) else 0.0,
            "custom_rules_applied": bool(self.custom_rules),
            "metadata_columns_used": self.metadata_columns if original_df is not None else [],
            "failed_batches": self.failed_batches,
            "failed_pairs": self.failed_pairs,
        }
        if self.failed_batches:
            warnings.warn(
                f"{self.failed_batches} LLM batch(es) failed ({self.failed_pairs} pairs); "
                f"on_failure='{self.on_failure}' was applied to them", DeduplixWarning, stacklevel=2)

        return ValidationResult(
            validated_pairs=validated_df,
            removed_pairs=removed_df,
            metadata=meta,
        )