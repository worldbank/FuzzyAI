import os
import re
import json
import time
from typing import Optional, List, Dict, Any,Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from .core import Validator, ValidationResult, MatchResult


class RuleBasedValidator(Validator):
    """Simple rule-based validation"""
    
    def __init__(self, rules: Optional[List[Callable]] = None, min_score: float = 90.0):
        self.min_score = min_score
        self.rules = rules or [self.default_rule]
    
    def default_rule(self, row: pd.Series) -> bool:
        """Default validation rule - accept high similarity scores"""
        return row['similarity_score'] >= self.min_score
    
    def validate(self, match_result: MatchResult, **kwargs) -> ValidationResult:
        """Apply rules to validate matches"""
        
        if match_result.pairs.empty:
            return ValidationResult(
                validated_pairs=pd.DataFrame(),
                removed_pairs=pd.DataFrame(),
                metadata={'message': 'No pairs to validate'}
            )
        
        validated = []
        removed = []
        
        for _, row in match_result.pairs.iterrows():
            if all(rule(row) for rule in self.rules):
                validated.append(row.to_dict())
            else:
                removed.append(row.to_dict())
        
        validated_df = pd.DataFrame(validated) if validated else pd.DataFrame()
        removed_df = pd.DataFrame(removed) if removed else pd.DataFrame()
        
        metadata = {
            'rules_applied': len(self.rules),
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
        *,
        databricks_host: Optional[str] = None,
        databricks_endpoint: Optional[str] = None,
        use_langchain_databricks: bool = False,
    ):
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

        # Databricks extras
        self.databricks_host = databricks_host or os.getenv("DATABRICKS_HOST")
        self.databricks_endpoint = databricks_endpoint
        self.use_langchain_databricks = use_langchain_databricks

        self.client = None
        self._client_mode = None  # "openai_v1" | "anthropic_v1" | "databricks_openai" | "databricks_langchain"
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
        # Prefer OpenAI-compatible Foundation Models API (no extra deps)
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

            # base_url points to /serving-endpoints (OpenAI-compatible)
            base_url = self.databricks_host.rstrip("/") + "/serving-endpoints"
            self.client = OpenAI(api_key=token, base_url=base_url)
            self._client_mode = "databricks_openai"
            return

        # Optional: LangChain ChatDatabricks
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

    def _create_prompt(self, batch: pd.DataFrame, original_df: pd.DataFrame = None) -> str:
        """
        Create prompt for LLM validation with optional metadata support.
        Works for both metadata and non-metadata cases.
        """
        
        # Build pair descriptions
        pairs_text = []
        for idx, row in batch.iterrows():
            pair_desc = f"Pair {idx}: '{row['name1']}' vs '{row['name2']}' (similarity: {row['similarity_score']:.1f}%)"
            
            # Add metadata if available
            if original_df is not None and self.metadata_columns:
                metadata_parts = []
                for col in self.metadata_columns:
                    if col in original_df.columns:
                        # Get metadata for both entities
                        df1_match = original_df[original_df['id'] == row['id1']]
                        df2_match = original_df[original_df['id'] == row['id2']]
                        
                        val1 = df1_match[col].iloc[0] if len(df1_match) > 0 else 'N/A'
                        val2 = df2_match[col].iloc[0] if len(df2_match) > 0 else 'N/A'
                        metadata_parts.append(f"{col}: {val1} vs {val2}")
                
                if metadata_parts:
                    pair_desc += f"\n  Metadata: {', '.join(metadata_parts)}"
            
            pairs_text.append(pair_desc)
        
        # Build base rules (generic, always applied)
        base_rules = [
            "- Carefully determine whether each pair represents the same entity or different entities",
            "- Consider all aspects of the names and any provided context"
        ]
        
        # Add custom instructions if provided
        if 'custom_instructions' in self.custom_rules:
            base_rules.append(f"- {self.custom_rules['custom_instructions']}")
        
        # Construct final prompt
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

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with multiple fallback strategies"""
        # Try fenced block first
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            return json.loads(fence.group(1))
        
        # Try first JSON object
        obj = re.search(r"\{.*\}", text, re.DOTALL)
        if obj:
            return json.loads(obj.group(0))
        
        # Try parsing as-is
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
            # For Databricks, `model` is the serving endpoint name (or override via databricks_endpoint)
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
            # LangChain ChatDatabricks
            result = self.client.invoke(prompt)
            # LangChain messages typically have `.content`
            return getattr(result, "content", str(result))

        if self._client_mode == "anthropic_v1":
            # Anthropic Messages API
            # Claude needs max_tokens; tune as you wish.
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=self.temperature,
                system="Identify duplicate entities. Reply with strict JSON only.",
                messages=[{"role": "user", "content": prompt}],
            )
            # Response content is a list of content blocks; extract text blocks.
            parts = []
            for block in getattr(resp, "content", []) or []:
                # block.type can be "text" etc.
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
                data = self._extract_json(out)
                decisions: Dict[int, Dict[str, Any]] = {}
                for d in data.get("decisions", []):
                    idx = d.get("pair_index")
                    if idx in batch.index:
                        decisions[idx] = {
                            "is_duplicate": bool(d.get("is_duplicate", True)),
                            "reason": d.get("reason", ""),
                        }
                return decisions
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Batch processing failed after {self.max_retries} attempts: {e}")
                    # Conservative fallback: keep all pairs
                    return {idx: {"is_duplicate": True, "reason": "LLM validation failed"}
                            for idx in batch.index}

    # ---------- Pre-filtering based on hard rules ----------

    def _apply_hard_rules(self, pairs_df: pd.DataFrame, original_df: pd.DataFrame) -> tuple[pd.DataFrame, List[Dict]]:
        """Apply hard rules to pre-filter pairs before LLM validation"""
        if original_df is None:
            return pairs_df, []
            
        filtered_pairs = []
        removed_by_rule = []
        
        for _, row in pairs_df.iterrows():
            should_remove = False
            removal_reason = ""
            
            # Country matching rule
            if self.custom_rules.get('require_same_country') and 'country' in original_df.columns:
                df1_match = original_df[original_df['id'] == row['id1']]
                df2_match = original_df[original_df['id'] == row['id2']]
                
                if len(df1_match) > 0 and len(df2_match) > 0:
                    country1 = df1_match['country'].iloc[0]
                    country2 = df2_match['country'].iloc[0]
                    
                    if pd.notna(country1) and pd.notna(country2) and country1 != country2:
                        should_remove = True
                        removal_reason = f'Different countries: {country1} vs {country2}'
            
            # Industry matching rule
            if not should_remove and self.custom_rules.get('require_same_industry') and 'industry' in original_df.columns:
                df1_match = original_df[original_df['id'] == row['id1']]
                df2_match = original_df[original_df['id'] == row['id2']]
                
                if len(df1_match) > 0 and len(df2_match) > 0:
                    industry1 = df1_match['industry'].iloc[0]
                    industry2 = df2_match['industry'].iloc[0]
                    
                    if pd.notna(industry1) and pd.notna(industry2) and industry1 != industry2:
                        should_remove = True
                        removal_reason = f'Different industries: {industry1} vs {industry2}'
            
            # Revenue difference rule
            if not should_remove and 'max_revenue_difference_percent' in self.custom_rules and 'revenue' in original_df.columns:
                df1_match = original_df[original_df['id'] == row['id1']]
                df2_match = original_df[original_df['id'] == row['id2']]
                
                if len(df1_match) > 0 and len(df2_match) > 0:
                    revenue1 = df1_match['revenue'].iloc[0]
                    revenue2 = df2_match['revenue'].iloc[0]
                    
                    if pd.notna(revenue1) and pd.notna(revenue2) and revenue1 > 0 and revenue2 > 0:
                        max_rev = max(revenue1, revenue2)
                        min_rev = min(revenue1, revenue2)
                        diff_percent = ((max_rev - min_rev) / max_rev) * 100
                        
                        if diff_percent > self.custom_rules['max_revenue_difference_percent']:
                            should_remove = True
                            removal_reason = f'Revenue difference too large: {diff_percent:.1f}% > {self.custom_rules["max_revenue_difference_percent"]}%'
            
            if should_remove:
                row_dict = row.to_dict()
                row_dict['validation_reason'] = removal_reason
                removed_by_rule.append(row_dict)
            else:
                filtered_pairs.append(row)
        
        if removed_by_rule:
            print(f"Pre-filtered {len(removed_by_rule)} pairs due to hard rules")
        
        return pd.DataFrame(filtered_pairs) if filtered_pairs else pd.DataFrame(), removed_by_rule

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
            Additional parameters (for compatibility)
        
        Returns
        -------
        ValidationResult
            Results of the validation process
        """
        pairs_df = match_result.pairs.copy()
        
        if pairs_df.empty:
            return ValidationResult(
                validated_pairs=pd.DataFrame(),
                removed_pairs=pd.DataFrame(),
                metadata={"message": "No pairs to validate"},
            )

        # Apply hard rules pre-filtering if configured
        removed_by_rules = []
        if self.custom_rules and original_df is not None:
            pairs_df, removed_by_rules = self._apply_hard_rules(pairs_df, original_df)
            
            if pairs_df.empty:
                # All pairs were filtered out by hard rules
                return ValidationResult(
                    validated_pairs=pd.DataFrame(),
                    removed_pairs=pd.DataFrame(removed_by_rules) if removed_by_rules else pd.DataFrame(),
                    metadata={
                        "message": "All pairs filtered by hard rules",
                        "model": self.model,
                        "provider": self.provider,
                        "removed_by_rules": len(removed_by_rules),
                    }
                )

        # Create batches for LLM processing
        batches = [
            pairs_df.iloc[i : i + self.batch_size]
            for i in range(0, len(pairs_df), self.batch_size)
        ]

        # Process batches in parallel
        all_decisions: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futures = {
                ex.submit(self._process_batch, batch, original_df): batch 
                for batch in batches
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM validation"):
                try:
                    all_decisions.update(fut.result())
                except Exception as e:
                    print(f"Error processing batch: {e}")

        # Collect results
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

        # Add hard rule removals to removed pairs
        removed_rows.extend(removed_by_rules)

        validated_df = pd.DataFrame(validated_rows) if validated_rows else pd.DataFrame()
        removed_df = pd.DataFrame(removed_rows) if removed_rows else pd.DataFrame()

        # Prepare metadata
        meta = {
            "model": self.model if self.provider != "databricks" else (self.databricks_endpoint or self.model),
            "provider": self.provider,
            "batches_processed": len(batches),
            "validated_count": len(validated_df),
            "removed_count": len(removed_df),
            "validation_rate": len(validated_df) / len(match_result.pairs) if len(match_result.pairs) else 0.0,
            "custom_rules_applied": bool(self.custom_rules),
            "metadata_columns_used": self.metadata_columns if original_df is not None else [],
        }

        return ValidationResult(
            validated_pairs=validated_df,
            removed_pairs=removed_df,
            metadata=meta,
        )