
"""
LLM-based consolidation for cross-dataset matching results.

Uses an LLM to intelligently select the best match when an entity has
multiple potential matches, considering metadata and custom instructions.
"""

import os
import re
import json
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm


@dataclass
class LLMConsolidationConfig:
    """
    Configuration for LLM-based consolidation of cross-dataset matches.
    
    Uses an LLM to select the "main" match when an entity has multiple candidates.
    The LLM considers metadata, similarity scores, and custom instructions.
    
    Parameters
    ----------
    enabled : bool
        Whether to apply LLM consolidation (default: True)
    api_key : Optional[str]
        API key for the LLM provider. If None, reads from environment
    model : str
        Model name (default: 'gpt-4o-mini')
    provider : str
        LLM provider: 'openai', 'anthropic', or 'databricks' (default: 'openai')
    temperature : float
        Sampling temperature (default: 0.0 for deterministic)
    max_retries : int
        Retry attempts per batch (default: 3)
    batch_size : int
        Number of entities to process per LLM call (default: 5)
        Lower = more calls but more focused decisions
    n_workers : int
        Parallel workers for processing (default: 4)
    checkpoint_every_n_batches : int
        Save progress every N batches (default: 5)
    metadata_columns : Dict[str, List[str]], optional
        Columns to include in LLM context from each dataframe:
        {
            'df1': ['column1', 'column2'],
            'df2': ['column3', 'column4']
        }
    instructions : str, optional
        Custom instructions for the LLM. Examples:
        - "Prefer matches from verified sources when quality is similar"
        - "Prioritize recent data over older data"
        - "Select the match with the most complete information"
    selection_criteria : List[str], optional
        List of criteria for selection. Will be formatted into instructions.
        Example: ['data_quality', 'completeness', 'recency']
    include_similarity_scores : bool
        Whether to show similarity scores to the LLM (default: True)
    other_candidates_column : str
        Column name for other candidate IDs (default: 'other_candidates')
    databricks_host : Optional[str]
        Databricks workspace URL (for provider='databricks')
    databricks_endpoint : Optional[str]
        Databricks serving endpoint name (for provider='databricks')
    
    Examples
    --------
    Basic LLM consolidation:
    >>> config = LLMConsolidationConfig(
    ...     enabled=True,
    ...     model='gpt-4o-mini',
    ...     instructions="Select the most complete and accurate match"
    ... )
    
    With metadata and custom criteria:
    >>> config = LLMConsolidationConfig(
    ...     enabled=True,
    ...     model='gpt-4o-mini',
    ...     metadata_columns={
    ...         'df1': ['entity_type', 'country'],
    ...         'df2': ['source_system', 'data_quality', 'last_updated']
    ...     },
    ...     instructions='''
    ...         Select the best match based on:
    ...         1. Prefer 'verified' data_quality over 'unverified'
    ...         2. Prefer more recent last_updated dates
    ...         3. Prefer matches from 'system_a' source when quality is similar
    ...         4. Consider the similarity score as a secondary factor
    ...     '''
    ... )
    
    With selection criteria (auto-formatted):
    >>> config = LLMConsolidationConfig(
    ...     enabled=True,
    ...     metadata_columns={'df2': ['data_quality', 'completeness_score']},
    ...     selection_criteria=[
    ...         'Higher data_quality (verified > unverified)',
    ...         'Higher completeness_score',
    ...         'Higher similarity score'
    ...     ]
    ... )
    """
    
    enabled: bool = True
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    temperature: float = 0.0
    max_retries: int = 3
    batch_size: int = 5
    n_workers: int = 4
    checkpoint_every_n_batches: int = 5
    metadata_columns: Optional[Dict[str, List[str]]] = None
    instructions: Optional[str] = None
    selection_criteria: Optional[List[str]] = None
    include_similarity_scores: bool = True
    other_candidates_column: str = 'other_candidates'
    databricks_host: Optional[str] = None
    databricks_endpoint: Optional[str] = None


class LLMConsolidationEngine:
    """Engine for LLM-based consolidation of matches"""
    
    def __init__(self, config: LLMConsolidationConfig):
        self.config = config
        self.client = None
        self._client_mode = None
        self._init_client()
    
    def _init_client(self):
        """Initialize the LLM client based on provider"""
        if self.config.provider == "openai":
            self._init_openai()
        elif self.config.provider == "anthropic":
            self._init_anthropic()
        elif self.config.provider == "databricks":
            self._init_databricks()
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
        except Exception as e:
            raise ImportError("Install openai>=1.0: pip install openai") from e
        
        key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY missing")
        self.client = OpenAI(api_key=key)
        self._client_mode = "openai"
    
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        try:
            from anthropic import Anthropic
        except Exception as e:
            raise ImportError("Install anthropic>=0.30: pip install anthropic") from e
        
        key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY missing")
        self.client = Anthropic(api_key=key)
        self._client_mode = "anthropic"
    
    def _init_databricks(self):
        """Initialize Databricks client"""
        try:
            from openai import OpenAI
        except Exception as e:
            raise ImportError("Install openai>=1.0: pip install openai") from e
        
        token = self.config.api_key or os.getenv("DATABRICKS_TOKEN") or os.getenv("DATABRICKS_API_KEY")
        host = self.config.databricks_host or os.getenv("DATABRICKS_HOST")
        
        if not token or not host:
            raise ValueError("Databricks credentials missing")
        
        base_url = host.rstrip("/") + "/serving-endpoints"
        self.client = OpenAI(api_key=token, base_url=base_url)
        self._client_mode = "databricks"
    
    def consolidate(
        self,
        cross_matches: pd.DataFrame,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        id_col1: str,
        id_col2: str,
        name_col2: str,
        checkpointer=None,
        data_hash: Optional[str] = None,
        resume: bool = True
    ) -> pd.DataFrame:
        """
        Apply LLM-based consolidation to select main match for each df1 entity.
        
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
        checkpointer : Checkpointer, optional
            Checkpointer instance for saving progress
        data_hash : str, optional
            Hash for checkpoint files
        resume : bool
            Whether to resume from checkpoint
            
        Returns
        -------
        pd.DataFrame
            Consolidated dataframe with one row per df1 entity
        """
        
        if not self.config.enabled or cross_matches.empty:
            result = df1.copy()
            result[f'main_{id_col2}'] = None
            result[f'main_{name_col2}'] = None
            result['main_similarity_score'] = None
            result[self.config.other_candidates_column] = None
            result['total_matches'] = 0
            result['llm_reasoning'] = None
            return result
        
        # Enrich matches with metadata
        enriched_matches = self._enrich_matches(cross_matches, df1, df2, id_col1, id_col2)
        
        # Group by df1 entity
        entity_groups = {}
        for df1_id in df1[id_col1].unique():
            entity_matches = enriched_matches[enriched_matches['df1_id'] == df1_id]
            if not entity_matches.empty:
                entity_groups[df1_id] = entity_matches
        
        # Process with LLM (with checkpointing)
        decisions = self._process_with_llm(
            entity_groups,
            df1,
            id_col1,
            checkpointer,
            data_hash,
            resume
        )
        
        # Build consolidated result
        consolidated_rows = []
        
        for df1_id in df1[id_col1].unique():
            df1_row = df1[df1[id_col1] == df1_id].iloc[0].to_dict()
            
            if df1_id in decisions:
                decision = decisions[df1_id]
                entity_matches = entity_groups.get(df1_id, pd.DataFrame())
                
                # Add main match info
                df1_row[f'main_{id_col2}'] = decision['selected_id']
                df1_row[f'main_{name_col2}'] = decision['selected_name']
                df1_row['main_similarity_score'] = decision['similarity_score']
                df1_row['llm_reasoning'] = decision.get('reasoning', '')
                
                # Other candidates
                if not entity_matches.empty:
                    other_ids = [
                        str(id_val) for id_val in entity_matches['df2_id'].tolist()
                        if id_val != decision['selected_id']
                    ]
                    df1_row[self.config.other_candidates_column] = ', '.join(other_ids) if other_ids else None
                    df1_row['total_matches'] = len(entity_matches)
                else:
                    df1_row[self.config.other_candidates_column] = None
                    df1_row['total_matches'] = 0
            else:
                # No matches
                df1_row[f'main_{id_col2}'] = None
                df1_row[f'main_{name_col2}'] = None
                df1_row['main_similarity_score'] = None
                df1_row['llm_reasoning'] = None
                df1_row[self.config.other_candidates_column] = None
                df1_row['total_matches'] = 0
            
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
        """Add metadata columns to matches"""
        enriched = matches.copy()
        
        # Add df1 metadata
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
        
        # Add df2 metadata
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
        
        return enriched
    
    def _create_prompt(
        self,
        batch_entities: Dict[Any, pd.DataFrame],
        df1: pd.DataFrame,
        id_col1: str
    ) -> str:
        """Create prompt for LLM to select best matches"""
        
        # Build instructions
        instructions = self.config.instructions or "Select the best match for each entity"
        
        if self.config.selection_criteria:
            criteria_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(self.config.selection_criteria))
            instructions += f"\n\nSelection criteria:\n{criteria_text}"
        
        # Build entity descriptions
        entities_text = []
        
        for entity_id, matches_df in batch_entities.items():
            # Get entity info from df1
            entity_row = df1[df1[id_col1] == entity_id].iloc[0]
            entity_name = entity_row.get('name', entity_row.get('entity_name', str(entity_id)))
            
            entity_desc = f"\nEntity: {entity_name} (ID: {entity_id})\n"
            entity_desc += "Candidates:\n"
            
            # List all candidates
            for idx, match_row in matches_df.iterrows():
                candidate_desc = f"  Candidate {idx}:\n"
                candidate_desc += f"    Name: {match_row['df2_name']}\n"
                candidate_desc += f"    ID: {match_row['df2_id']}\n"
                
                if self.config.include_similarity_scores:
                    candidate_desc += f"    Similarity Score: {match_row['similarity_score']:.1f}%\n"
                
                # Add metadata
                if self.config.metadata_columns:
                    if 'df2' in self.config.metadata_columns:
                        for col in self.config.metadata_columns['df2']:
                            if col in match_row:
                                candidate_desc += f"    {col}: {match_row[col]}\n"
                
                entity_desc += candidate_desc
            
            entities_text.append(entity_desc)
        
        # Construct full prompt
        prompt = f"""You are helping to consolidate duplicate entity matches. 
    For each entity below, select the BEST candidate match from the options provided.

    {instructions}

    {''.join(entities_text)}

    Respond with ONLY a JSON object in this format:
    {{
    "selections": [
        {{
        "entity_id": <entity_id>,
        "selected_candidate_index": <index>,
        "selected_candidate_id": <id>,
        "reasoning": "Brief explanation of why this candidate was selected"
        }},
        ...
    ]
    }}

    DO NOT include any text outside the JSON object. Return only valid JSON."""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Make LLM API call"""
        if self._client_mode == "openai":
            resp = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a data matching expert. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature
            )
            return resp.choices[0].message.content
        
        elif self._client_mode == "databricks":
            model = self.config.databricks_endpoint or self.config.model
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a data matching expert. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature
            )
            return resp.choices[0].message.content
        
        elif self._client_mode == "anthropic":
            resp = self.client.messages.create(
                model=self.config.model,
                max_tokens=2000,
                temperature=self.config.temperature,
                system="You are a data matching expert. Respond only with valid JSON.",
                messages=[{"role": "user", "content": prompt}]
            )
            parts = []
            for block in getattr(resp, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts).strip()
        
        raise RuntimeError("LLM client not initialized")
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        # Try fenced block
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            return json.loads(fence.group(1))
        
        # Try first JSON object
        obj = re.search(r"\{.*\}", text, re.DOTALL)
        if obj:
            return json.loads(obj.group(0))
        
        # Try parsing as-is
        return json.loads(text)
    
    def _process_batch(
        self,
        batch_entities: Dict[Any, pd.DataFrame],
        df1: pd.DataFrame,
        id_col1: str
    ) -> Dict[Any, Dict[str, Any]]:
        """Process a batch of entities with LLM"""
        
        prompt = self._create_prompt(batch_entities, df1, id_col1)
        
        for attempt in range(self.config.max_retries):
            try:
                response = self._call_llm(prompt)
                data = self._extract_json(response)
                
                # Parse selections
                decisions = {}
                for selection in data.get("selections", []):
                    entity_id = selection.get("entity_id")
                    candidate_idx = selection.get("selected_candidate_index")
                    candidate_id = selection.get("selected_candidate_id")
                    reasoning = selection.get("reasoning", "")
                    
                    if entity_id in batch_entities:
                        matches_df = batch_entities[entity_id]
                        
                        # Find the selected match
                        if candidate_id is not None:
                            selected_match = matches_df[matches_df['df2_id'] == candidate_id]
                        elif candidate_idx is not None and candidate_idx in matches_df.index:
                            selected_match = matches_df.loc[[candidate_idx]]
                        else:
                            # Fallback: first match
                            selected_match = matches_df.iloc[:1]
                        
                        if not selected_match.empty:
                            match_row = selected_match.iloc[0]
                            decisions[entity_id] = {
                                'selected_id': match_row['df2_id'],
                                'selected_name': match_row['df2_name'],
                                'similarity_score': match_row['similarity_score'],
                                'reasoning': reasoning
                            }
                
                return decisions
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"Batch processing failed: {e}")
                    # Fallback: select highest score for each
                    fallback_decisions = {}
                    for entity_id, matches_df in batch_entities.items():
                        best_match = matches_df.loc[matches_df['similarity_score'].idxmax()]
                        fallback_decisions[entity_id] = {
                            'selected_id': best_match['df2_id'],
                            'selected_name': best_match['df2_name'],
                            'similarity_score': best_match['similarity_score'],
                            'reasoning': 'LLM failed, selected highest score'
                        }
                    return fallback_decisions
    
    def _process_with_llm(
        self,
        entity_groups: Dict[Any, pd.DataFrame],
        df1: pd.DataFrame,
        id_col1: str,
        checkpointer,
        data_hash: Optional[str],
        resume: bool
    ) -> Dict[Any, Dict[str, Any]]:
        """Process all entities with LLM, with checkpointing"""
        
        # Create batches
        entity_ids = list(entity_groups.keys())
        batches = []
        for i in range(0, len(entity_ids), self.config.batch_size):
            batch_ids = entity_ids[i:i + self.config.batch_size]
            batch_entities = {eid: entity_groups[eid] for eid in batch_ids}
            batches.append(batch_entities)
        
        # Load checkpoint if resuming
        all_decisions = {}
        processed_batches = set()
        
        if self.config.checkpoint_every_n_batches > 0 and checkpointer and data_hash and resume:
            checkpoint_data = checkpointer.load('llm_consolidation_progress', data_hash)
            if checkpoint_data is not None:
                for _, row in checkpoint_data.iterrows():
                    entity_id = row['entity_id']
                    all_decisions[entity_id] = {
                        'selected_id': row['selected_id'],
                        'selected_name': row['selected_name'],
                        'similarity_score': row['similarity_score'],
                        'reasoning': row.get('reasoning', '')
                    }
                
                # Determine processed batches
                processed_ids = set(all_decisions.keys())
                for batch_idx, batch in enumerate(batches):
                    if all(eid in processed_ids for eid in batch.keys()):
                        processed_batches.add(batch_idx)
                
                print(f"  Resuming from checkpoint: {len(processed_batches)}/{len(batches)} batches processed")
        
        # Process batches
        batches_to_process = [(i, batch) for i, batch in enumerate(batches) if i not in processed_batches]
        
        if batches_to_process:
            batch_counter = len(processed_batches)
            
            with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
                futures = {
                    executor.submit(self._process_batch, batch, df1, id_col1): (batch_idx, batch)
                    for batch_idx, batch in batches_to_process
                }
                
                for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM consolidation"):
                    try:
                        batch_decisions = fut.result()
                        all_decisions.update(batch_decisions)
                        batch_counter += 1
                        
                        # Checkpoint
                        if (self.config.checkpoint_every_n_batches > 0 and
                            checkpointer and data_hash and
                            batch_counter % self.config.checkpoint_every_n_batches == 0):
                            
                            progress_rows = []
                            for eid, decision in all_decisions.items():
                                progress_rows.append({
                                    'entity_id': eid,
                                    'selected_id': decision['selected_id'],
                                    'selected_name': decision['selected_name'],
                                    'similarity_score': decision['similarity_score'],
                                    'reasoning': decision.get('reasoning', '')
                                })
                            progress_df = pd.DataFrame(progress_rows)
                            checkpointer.save(progress_df, 'llm_consolidation_progress', data_hash)
                            print(f"  Checkpoint saved: {batch_counter}/{len(batches)} batches processed")
                    
                    except Exception as e:
                        print(f"Error processing batch: {e}")
        
        return all_decisions