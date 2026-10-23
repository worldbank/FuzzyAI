# FuzzyAI modular pipeline — implementation TODO

Oct 23, 2026 · @A

## Scope and compatibility contract

The refactor adds a staged, pluggable engine under `deduplix/engine/` and keeps every public name, signature, default and output column of 0.1.0 working. New behaviour is opt-in; the only behaviour changes allowed are the bug fixes listed in Phase 9, each called out in the changelog.

**Goals**

- Every stage (normalize, block, compare, score, route, validate, cluster, consolidate) is swappable three ways: by name in YAML, by a registered function, or by a Python object.
- Cost scales with candidate pairs, not n×m, through blocking.
- The LLM layer is split into provider, prompt template, output parser and decision policy.
- Every decision is traceable to one row of the pairs table.

**Non-goals for 0.2:** Spark backend, an online matching service, a UI. The design must not block them (pairs-table contract, fitted blockers kept in `ctx.artifacts`).

**Order of work:** Phase 0 → Phase 1 → Phases 2–6 in parallel → Phase 7 → Phases 8–10. Phase 9 fixes can land any time after Phase 0.

**What must not change (verified by the Phase 0 snapshot tests)**

| Surface | Contract |
| --- | --- |
| Imports | The 10 names in `deduplix.__all__`, and the module paths `deduplix.core`, `.matchers`, `.validators`, `.consolidation`, `.llm_consolidation`, `.checkpointing`, `.exceptions`, `.validation`, `.security`, `.utils` |
| `DeduplicationPipeline(...)` | Existing positional order kept; new keyword arguments only appended |
| `run(df, id_column, name_column, additional_columns, resume)` | Same signature, returns `DeduplicationResult` |
| `run_cross_dataset(...)` | Same 11 parameters, returns `CrossDatasetResult` |
| Pair columns | `id1, id2, name1, name2, similarity_score`; cross adds `matched_column1, matched_column2, source1, source2`; validation adds `validation_reason` |
| Result columns | `entity_groups`: `entity_id, entity_name, group_id` (0 = singleton); `cross_matches`: `df1_id, df1_name, df2_id, df2_name, similarity_score, validation_reason` (+ matched columns) |
| Score scale | `similarity_score` stays a 0–100 float |
| Checkpoint stage names | `matching`, `validation`, `cross_matching`, `cross_validation`, `validation_progress`, so existing runs still resume |
| Constructors | `FuzzyMatcher`, `RuleBasedValidator`, `LLMValidator`, `ConsolidationConfig`, `LLMConsolidationConfig`: all arguments kept, same defaults |
| Saved files | `entity_groups.csv`, `duplicate_pairs.csv`, `statistics.json`, `cross_matches.csv`, `metadata.json` |
| CLI | `run`, `init`, `analyze`, `show-group`, `remove` with their current options |

Where the README or tests disagree with the code, the README and tests define the contract, because that is what users wrote code against. Phase 0 lists those gaps.

## Module layout, dependencies, conventions

New code lives in new sub-packages; the 0.1.0 modules stay where they are and become thin facades in Phase 7.

```text
deduplix/
  __init__.py            # existing exports + Pipeline, PipelineResult, register, load_pipeline, EngineMatcher
  core.py matchers.py validators.py consolidation.py llm_consolidation.py   # legacy, kept
  engine/
    schema.py            # column-name constants, validate_pairs(), to_legacy_pairs()
    protocols.py         # Normalizer, Blocker, Comparator, Scorer, Router, Validator, Clusterer, Consolidator, Hook
    registry.py          # register(), build(), list_components(), entry points
    context.py           # RunContext
    pipeline.py          # Pipeline, PipelineResult
    router.py            # ThresholdRouter, RuleRouter
    config.py            # pydantic models, load_pipeline()
    hooks.py             # LoggingHook, ProgressHook, CheckpointHook, MLflowHook
  normalize/   text.py name_parser.py tables/{legal_forms,designators,abbreviations,stopwords}.yaml
  blocking/    base.py exact.py token_index.py tfidf.py phonetic.py minhash.py embedding.py brute_force.py union.py metrics.py
  compare/     string.py fields.py designators.py grades.py
  scoring/     weighted.py feature.py grade_table.py fellegi_sunter.py model.py
  llm/         providers/{base,openai,anthropic,databricks,callable,fake}.py templates.py parsers.py policy.py cache.py validator.py prompts/*.j2
  cluster/     components.py guarded.py louvain.py report.py
  evaluation/  metrics.py silver.py synthetic.py train.py
benchmarks/run.py
tests/{compat,golden,engine,normalize,blocking,compare,scoring,llm,cluster,evaluation}/
```

**Dependencies**

| Install | Packages | Why |
| --- | --- | --- |
| core | pandas, numpy, rapidfuzz>=3.6, networkx>=2.8, pydantic>=2, pyyaml, tqdm, click, jinja2 | rapidfuzz 3.6 adds `process.cpdist` (pairwise, no matrix); networkx 2.8 adds Louvain |
| `[blocking]` | scikit-learn>=1.3, scipy>=1.10, sparse\_dot\_topn>=1.1 | TF-IDF and token-index blockers, sparse top-k matmul |
| `[phonetic]` | metaphone>=0.6 | `doublemetaphone()` |
| `[minhash]` | datasketch>=1.6 | MinHash LSH |
| `[embeddings]` | sentence-transformers>=2.7, hnswlib>=0.8; optional faiss-cpu>=1.8 | encoder and ANN index |
| `[normalize]` | anyascii>=0.3, tldextract>=5 | transliteration, registrable domain |
| `[llm]` | openai>=1.0, anthropic>=0.30 | providers (current extra pins openai>=0.27, which the code cannot use) |
| `[eval]` | scikit-learn, psutil | GBDT scorer, calibration, benchmark memory |
| `[all]` | everything above |  |

Licence rule: the library is MIT, so no GPL dependency, even optional. That excludes Unidecode (use anyascii, ISC) and leidenalg/python-igraph (use networkx Louvain, BSD).

**Conventions**

- `from __future__ import annotations` in every new module; `typing.Protocol` with `@runtime_checkable`.
- Stages never mutate their inputs; they return new frames or new columns.
- Internal pair keys are row positions `i1`, `i2` (int32 when both tables are under 2^31 rows, else int64). User ids are attached only when building outputs.
- Engine code logs through `logging.getLogger("deduplix.<module>")`; no `print`. Legacy classes keep their prints.
- Every randomised component takes `random_state` (default 0) so runs are reproducible.
- Missing values stay missing (`pd.NA`) through features; a comparator never turns a missing side into 0.
- Black at 100 columns, isort, mypy on `deduplix/engine` and new packages, numpy-style docstrings.
- Tests never hit the network; LLM tests use `FakeProvider`.

## Phase 0 — Baseline and golden tests

Pin today's behaviour before changing code, so the facade can be proven equivalent. Reading the tests against the code shows several tests cannot pass on `master` today; confirm by running them.

- [ ] Branch `feature/modular-pipeline` from `master` (0.1.0, 26 commits).
- [ ] `pip install -e .[dev]` with pydantic 2, run `pytest -q`, write results to `docs/baseline.md`.
- [ ] Expected failures from reading the code (verify each):
  - `test_pipeline_with_file_checkpointing`, `test_pipeline_with_database_checkpointing`: `DeduplicationPipeline.__init__` (core.py L376–386) has no `checkpoint_type` or `checkpoint_db_path`, and no `checkpoint_type` attribute.
  - `test_pipeline_invalid_checkpoint_type` expects `ConfigurationError("Unsupported checkpoint type ...")`.
  - `test_run_with_validation_errors` expects `DataValidationError`; `run()` raises `ValueError` (core.py L465).
  - `test_compute_data_hash` expects a name change to change the hash; `_compute_data_hash` hashes only the id column and shape (core.py L390).
  - `test_matcher_creation_default` expects `n_workers == 4`; the default is 1 (matchers.py L83).
- [ ] Write `tests/data/make_synthetic.py` (seed 42) producing: 2,000 companies with injected variants, and a 1,000 × 1,500 cross-dataset pair. Reuse it in Phase 8.
- [ ] Golden fixtures in `tests/golden/`, generated from `master` by `scripts/make_golden.py` and stored as parquet:
  - matcher outputs for scorers `token_sort_ratio`, `token_set_ratio`, `WRatio`; punctuation removal on and off; multi-column cross matching.
  - `entity_groups` with group ids relabelled by the smallest entity id in each group, because networkx component order is not guaranteed.
  - pairs sorted by `(id1, id2)`; scores compared with `rtol=1e-6`.
- [ ] LLM golden tests: a `FakeProvider` replays fixed JSON responses; the prompts the legacy `LLMValidator` builds are stored as snapshot strings, so any wording change is deliberate.
- [ ] `tests/compat/test_public_api.py`: asserts the `deduplix.__all__` set and the `inspect.signature` of every public callable (parameter names, order, defaults) against `tests/compat/api_snapshot.json`. The snapshot changes only through `scripts/update_api_snapshot.py`, and only by appending parameters.
- [ ] GitHub Actions: Python 3.9–3.12, `pytest -m "not slow"`, plus a job with only core dependencies installed to prove optional extras stay optional. Leave `azure-pipelines-mirror.yml` untouched.

## Phase 1 — Core engine

The engine runs a fixed sequence of stages over one pairs table. Each stage reads columns and appends its own, so any stage can be replaced without touching the others.

&#91;embedded content: engine stages · 8 stages, 3 routes\]

The router settles clear accepts and rejects for free; only the review band costs validator calls, and the pairs table carries every decision into clustering.

**Pairs table** (`engine/schema.py` holds the names as constants)

| Column | dtype | Written by | Meaning |
| --- | --- | --- | --- |
| `i1`, `i2` | int32 or int64 | Block | row positions in left and right; dedup mode keeps only `i1 < i2` |
| `blk_<name>` | bool | Block | blocker `<name>` proposed this pair |
| `blk_score_<name>` | float32 | Block | that blocker's own similarity, when it has one |
| `f_<name>` | float32 or category | Compare | one feature |
| `score` | float32, 0–1 | Score | match score |
| `route` | category: accept, review, reject | Route |  |
| `decision` | category: match, non\_match, unvalidated | Route, Validate | final decision |
| `decision_prob` | float32 | Validate | optional probability |
| `decision_reason` | string | Validate |  |
| `decided_by` | string | Route, Validate | name of the stage that decided |

Records get `cluster_id` (int64) from the clusterer. `to_legacy_pairs()` builds `id1, id2, name1, name2, similarity_score`; `similarity_score = 100 × score` unless `legacy_score_feature` names an `f_` column to use instead.

**Protocols** (`engine/protocols.py`)

```python
@runtime_checkable
class Normalizer(Protocol):
    def __call__(self, records: pd.DataFrame, ctx: RunContext) -> pd.DataFrame: ...   # returns records + new columns

@runtime_checkable
class Blocker(Protocol):
    name: str
    def fit(self, right: pd.DataFrame, ctx: RunContext) -> "Blocker": ...
    def candidates(self, left: pd.DataFrame, ctx: RunContext) -> pd.DataFrame: ...  # i1, i2[, score]

@runtime_checkable
class Comparator(Protocol):
    def __call__(self, pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame,
                 ctx: RunContext) -> pd.DataFrame: ...                              # only new f_ columns, same index

class Scorer(Protocol):
    def __call__(self, pairs: pd.DataFrame, ctx: RunContext) -> pd.Series: ...      # float in [0, 1]

class Router(Protocol):
    def __call__(self, pairs: pd.DataFrame, ctx: RunContext) -> pd.Series: ...      # route category

class Validator(Protocol):
    name: str
    scope: Literal["review", "all", "accepted"]
    def __call__(self, pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame,
                 ctx: RunContext) -> pd.DataFrame: ...   # decision, decision_prob, decision_reason for rows it decided

class Clusterer(Protocol):
    def __call__(self, pairs: pd.DataFrame, n_records: int, ctx: RunContext) -> np.ndarray: ...  # cluster id per record

class Hook(Protocol):
    def on_run_start(self, ctx): ...
    def on_stage_start(self, stage: str, ctx, pairs) -> pd.DataFrame | None: ...   # non-None = cached result, stage skipped
    def on_stage_end(self, stage: str, ctx, pairs, seconds: float): ...
    def on_run_end(self, ctx, result): ...
    def on_error(self, stage: str, ctx, exc: BaseException): ...
```

Plain functions with these signatures are valid components; nothing has to subclass.

**Tasks**

- [ ] `engine/context.py`: dataclass `RunContext(left, right, mode: "dedup" | "link", columns: ColumnMap, random_state=0, artifacts: dict, checkpointer=None, logger)`. Both tables are reset to a RangeIndex on entry; in dedup mode `right is left`.
- [ ] `engine/schema.py`: column constants, `validate_pairs(df, stage)` (dtype and duplicate-column checks, called after every stage in debug mode), `to_legacy_pairs()`, `to_legacy_cross()`.
- [ ] `engine/registry.py`:
  - `_REGISTRY: dict[str, dict[str, Callable]]` with kinds `normalizer, blocker, comparator, scorer, router, validator, clusterer, consolidator, provider, parser, hook, encoder`.
  - `register(kind, name, *, overwrite=False)` decorator for classes and functions; a duplicate name raises `ConfigurationError` unless `overwrite=True`.
  - `build(kind, spec)`: spec is a string (no parameters), a one-key dict `{name: params}`, or an object. An object that satisfies the protocol is returned as is; a registered function with params becomes `functools.partial(fn, **params)`; a class is instantiated with `**params`.
  - Entry points loaded lazily on first `build()` from group `deduplix.<kind>s`, so other packages can add components without forking.
  - `list_components(kind=None)` returns name, kind, first docstring line, parameters.
- [ ] `engine/pipeline.py`: class `Pipeline(normalizers=(), blockers=(), comparators=(), scorer=None, router=None, validators=(), clusterer=None, consolidator=None, hooks=(), *, legacy_score_feature=None, max_pairs=50_000_000, undecided="unvalidated", random_state=0)` with `dedupe(df, id_column="id", name_column="name", **columns)`, `link(left, right, ...)`, `from_config(path_or_dict)`, `default()`, `to_config()`.
- [ ] Execution order in `_execute(ctx)`, each step wrapped by `_stage(name, fn)` which calls the hooks and times it:
  1. Normalize: apply normalizers in order to left, then to right in link mode.
  2. Block: `fit` each blocker on right, `candidates` on left, then `blocking.union.union_candidates`. Raise `ConfigurationError` when the union exceeds `max_pairs`, naming the blocker that contributed most.
  3. Compare: concatenate each comparator's columns; a column-name collision raises `ConfigurationError`.
  4. Score: the scorer's output becomes `score`. With no scorer and exactly one feature, `score` is that feature.
  5. Route: accept → `decision=match`, reject → `non_match`, review → undecided. No router means everything is review.
  6. Validate: validators run in order; each receives only the rows in its `scope` (default: undecided rows). Rows a validator cannot decide stay undecided for the next one.
  7. Undecided rows left at the end get `decision = undecided` (default `"unvalidated"`; `"match"` or `"non_match"` allowed).
  8. Cluster (dedup mode) on `decision == match` edges.
  9. Consolidate (link mode) when configured.
- [ ] `PipelineResult`: `.records`, `.pairs`, `.matches`, `.review`, `.clusters`, `.stats` (per-stage rows in and out, seconds), `.config` (resolved), `.explain(id1, id2)` (one pair with all columns), `.save(dir)` (parquet + `pipeline.yaml` + `stats.json`), `.to_legacy()` → `DeduplicationResult` or `CrossDatasetResult`.
- [ ] `engine/router.py`:
  - `ThresholdRouter(accept_above=0.95, reject_below=0.60, on="score")`: `>= accept_above` accept, `< reject_below` reject, otherwise review.
  - `RuleRouter(rules=[...], then=ThresholdRouter(...))`: rules run first. In YAML a rule is declarative (`{feature: f_designators, equals: reject, route: reject}`, also `in`, `gte`, `lt`, `isna`); in Python a rule may be a callable `pairs -> bool mask`. No string `eval`/`DataFrame.query` from config files.
- [ ] `engine/hooks.py`:
  - `LoggingHook` (on by default) logs rows in and out and seconds per stage.
  - `ProgressHook` wraps tqdm.
  - `CheckpointHook(checkpointer)` reuses `FileCheckpointer`/`DatabaseCheckpointer`. Key = sha256 of the stage name, the resolved config of that stage and every upstream stage, and the data hash, so changing a threshold invalidates that stage and the ones after it only.
  - `MLflowHook` (optional import) logs the flattened config as params, `stats` as metrics, and a 1,000-row sample of pairs as an artifact.
- [ ] `engine/config.py`: pydantic v2 models with `extra="forbid"` mirroring the stages. List entries are `name` or `{name: params}`; `compare` is a mapping `feature_name → {fn, left, right, ...}`. `${ENV_VAR}` interpolation is allowed only inside `provider` blocks (secrets). `load_pipeline(path)` returns a `Pipeline`; `Pipeline.to_config()` round-trips components built from the registry and writes `"<python object>"` with a warning for the rest.
- [ ] Tests: registry (register, duplicate, entry point via a fake distribution), `build` for all three spec forms, stage order, scope filtering, undecided policy, column-collision error, `max_pairs` guard, config round-trip, checkpoint invalidation when one threshold changes.

## Phase 2 — Normalization and name parsing

Names are parsed into parts instead of stripped: blocking uses the aggressive base name, while scoring keeps the legal form, fund numbers and qualifiers as features. All language-specific knowledge lives in YAML tables that users override per dataset.

**Output columns** (originals are never overwritten)

| Column | Example for "ABC Fund IV (Mauritius), L.P." |
| --- | --- |
| `name_norm` | `abc fund iv mauritius l p` |
| `name_base` | `abc` |
| `legal_form` / `legal_form_raw` | `LP` / `L.P.` |
| `designator_nums` | `4` (sorted, pipe-joined string so it survives parquet) |
| `designator_series` | empty (set for "Series B") |
| `vintage_year` | empty (set for "ABC Fund 2019") |
| `designator_quals` | empty (set for feeder, GP, management, ...) |
| `geo_qualifier` | `MU` |

**Tasks**

- [ ] `normalize/text.py`, each registered as a normalizer factory with `columns` (default: the name column) and `output` (default `name_norm`):
  - `lowercase`, `strip_whitespace`, `remove_punctuation(pattern=r"[^\w\s]")`, `ampersand` (`&` → `and`).
  - `fold_accents`: `unicodedata.normalize("NFKD")`, drop combining marks. No dependency.
  - `transliterate`: `anyascii` for non-Latin scripts (extra `[normalize]`).
  - `expand_abbreviations(table="abbreviations.yaml")`: whole-token replacement (intl → international, mfg → manufacturing, bros → brothers).
  - `chain` helper: `default_text = [strip_whitespace, fold_accents, lowercase, ampersand, remove_punctuation]`, matching legacy preprocessing when only `lowercase` and `strip_whitespace` are on.
- [ ] `normalize/tables/legal_forms.yaml` format:

```yaml
- code: LTD
  variants: [ltd, "ltd.", limited]
  countries: []          # empty = any country
  position: suffix       # suffix | prefix | any
- code: SADECV
  variants: ["s.a. de c.v.", "sa de cv"]
  countries: [MX]
- code: SABDECV
  variants: ["s.a.b. de c.v.", "sab de cv"]
  countries: [MX]
```

- [ ] Seed table: at least LTD, PLC, LLC, LLP, LP, INC, CORP, CO, GMBH, AG, KG, SA, SAS, SARL, SADECV, SABDECV, SPA, SRL, BV, NV, AB, AS, OY, PTY, PTE, BHD, SDN, KK, PVT, LIMITADA, EIRELI. Mark each with its countries.
- [ ] Matching rule: compile variants into one regex sorted longest first, applied to the token sequence at the end of the name only (`position: suffix`). This keeps "Limited Brands" intact. When the record has a country and the form lists countries, match only if the country is listed; unknown country matches any.
- [ ] `normalize/tables/designators.yaml`:

```yaml
prefix_words: [fund, series, tranche, class, partners, investors, capital, holdings, opportunities, vehicle]
roman_max: 30
roman_trailing: true        # a roman numeral as the last token counts even without a prefix word
year_range: [1950, 2035]    # 4-digit numbers in range go to vintage_year, not designator_nums
series_letters: true        # single letter after a prefix word -> designator_series
qualifiers:
  feeder: [feeder]
  master: [master]
  parallel: [parallel]
  offshore: [offshore]
  onshore: [onshore]
  coinvest: [co-invest, coinvest, co-investment, coinvestment]
  gp: [gp, general partner]
  management: [management, manager, advisors, advisers]
  blocker: [blocker]
  aiv: [aiv]
  sidecar: [sidecar]
```

- [ ] Designator rules in `name_parser`, in this order:
  1. Digits of length ≤ 4: in `year_range` → `vintage_year`, else → `designator_nums`.
  2. Roman numeral tokens (`i` to `roman_max`, parsed with a strict regex) count only after a prefix word or as the last token. This avoids "mix" or "civil".
  3. After a prefix word, a single letter that is not a roman numeral → `designator_series`. "Fund I" therefore reads as 1, not letter I.
  4. Qualifier phrases are matched as whole tokens, multi-word first.
- [ ] `name_parser(tables={...}, merge="extend" | "replace", country_column=None)`: user tables extend the defaults unless `merge="replace"`. Vectorised with pandas `.str` and precompiled regex; no per-row Python loop over more than the token list.
- [ ] Performance budget: under 10 s for 1M names on one core, measured in Phase 8.
- [ ] Tests: at least 60 parametrised cases, including:
  - "ABC Fund IV, L.P." → base `abc`, LP, nums 4.
  - "Mix Capital Ltd" → no designator.
  - "ABC Capital Management LLC" → qualifier `management`.
  - "ABC Fund 2019" → vintage 2019, no nums.
  - "Grupo Bimbo, S.A.B. de C.V." with country MX → SABDECV.
  - "Limited Brands Inc" → INC, base keeps "limited brands".
  - "ABC Fund I" vs "ABC Fund 1" → both nums 1.

## Phase 3 — Blockers

Blockers trade a little recall for a large cut in pairs; each one is measured by pair completeness against brute force before it becomes a default. All blockers share these parameters, implemented once in `blocking/base.py`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `name` | registry name | provenance label: `blk_<name>`, `blk_score_<name>` |
| `on` | `name_base`, else `name_norm`, else the name column | column the blocker reads |
| `partition_by` | none | candidates only between records with equal values (for example country) |
| `null_partition` | `"compare_all"` | records with a null partition value: compare with every partition, or `"skip"` |
| `top_k` | blocker-specific | maximum candidates kept per left record |

**Tasks**

- [ ] `BaseBlocker`: handles `partition_by` by looping over partition values (fit and query per partition, offsets mapped back to global positions), handles dedup mode (drop `i1 >= i2` after swapping so `i1 < i2`), and returns int arrays, never id columns.
- [ ] `exact` — `ExactKeyBlocker(keys=[...], normalizers={key: kind}, max_key_freq=50)`:
  - Key normalizers: `alnum` (strip non-alphanumerics, upper case, strip leading zeros), `lei` (20 characters, ISO 17442 check digits with ISO 7064 MOD 97-10; invalid becomes null), `domain` (tldextract registered domain, lower case, `www` removed), `phone` (E.164 if `phonenumbers` is installed).
  - Join with `merge` on each key; drop nulls.
  - Skip key values occurring more than `max_key_freq` times ("N/A", "000000") and log them.
- [ ] `token_index` — `TokenIndexBlocker(min_token_len=2, max_df=1000, max_df_ratio=None, stopwords="stopwords.yaml", top_k=50, min_score=0.0)`:
  - `CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b")` fitted on right; tokens with document frequency above `max_df` removed from the vocabulary and stored in `ctx.artifacts["token_index.dropped"]` with counts.
  - Score = sum of IDF (`log(N/df)`) over shared tokens: `(X_left · diag(idf)) @ X_right.T`, computed with `sp_matmul_topn(..., top_n=top_k, threshold=min_score)`.
  - Why the cap: a token in d records generates about d²/2 pairs.
- [ ] `tfidf_char` — `TfidfCharBlocker(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True, dtype=np.float32, top_n=20, threshold=0.5, chunk_size=10_000, n_threads=-1)`:
  - Fit the vectorizer on left ∪ right in link mode (IDF reflects both), on left in dedup mode.
  - Keep `B = X_right.T.tocsr()` from `fit`; for each chunk of left rows call `sp_matmul_topn(A_chunk, B, top_n=top_n, threshold=threshold, n_threads=n_threads)`.
  - Dedup mode: ask for `top_n + 1` and drop the diagonal.
  - Cosine goes to `blk_score_tfidf_char`.
- [ ] `phonetic` — `PhoneticBlocker(algorithm="double_metaphone", tokens="rarest", n=2, max_block_size=500)`:
  - `metaphone.doublemetaphone(token)` gives primary and secondary codes; both become keys.
  - `tokens="rarest"` encodes the `n` lowest-frequency tokens; `"first"` the first `n`.
  - Self-merge on (key, partition); blocks larger than `max_block_size` are dropped and logged.
  - Document as Latin-script and English-biased; never a default blocker.
- [ ] `minhash` — `MinHashLSHBlocker(shingle="char", k=3, num_perm=128, threshold=0.5, seed=1, verify=True)`:
  - datasketch `MinHash` per record (built with `MinHash.bulk`), stored as `LeanMinHash`; `MinHashLSH(threshold, num_perm)`; insert right, query left.
  - `verify=True` recomputes the estimated Jaccard and drops pairs below `threshold`.
  - Intended for incremental or online use (insert and query one record at a time); document that TF-IDF is faster in batch.
- [ ] `embedding` — `EmbeddingANNBlocker(encoder="sentence_transformers", model="intfloat/multilingual-e5-small", text_template="{name_base} {city} {country}", prefix="query: ", batch_size=256, index="hnsw", hnsw={M: 32, ef_construction: 200, ef: 100}, top_k=20, min_cosine=0.80, cache_dir=None)`:
  - Encoder protocol `encode(texts: list[str]) -> np.ndarray[float32]`, normalised to unit length. Built-ins: `sentence_transformers`, `databricks_embeddings` (calls a serving endpoint), `callable`.
  - Index: hnswlib `space="cosine"` by default; `index="faiss_ivfpq"` with `nlist`, `m`, `nbits` for tens of millions of rows.
  - Cache embeddings to `cache_dir/<sha1(model + text)>.npy` so reruns do not re-encode.
  - Document the failure mode: neighbours like "National Bank of Kenya" / "... of Egypt" and "Fund I" / "Fund IV" sit close; precision comes from later stages.
- [ ] `brute_force` — `BruteForceBlocker(scorer="token_sort_ratio", threshold=80.0, batch_size=500, n_workers=4)`: the legacy `cdist` loop moved here unchanged, returning `i1, i2, score/100`. Default for fewer than 20,000 records.
- [ ] `blocking/union.py` — `union_candidates(frames, n_right, mode)`:
  - Encode each pair as one int64 key `i1 * n_right + i2`; `np.unique` over the concatenation; scatter provenance flags and max scores back. Avoids a pandas groupby over 10^8 rows.
  - Optional `max_pairs_per_record`: keep the pairs with the highest max blocker score.
- [ ] `blocking/metrics.py`:
  - `reduction_ratio(n_candidates, n_left, n_right, mode)`.
  - `pair_completeness(candidates, gold)`, `pairs_quality(candidates, gold)`.
  - `blocking_report(candidates, gold)`: one row per blocker with candidates, PC, PQ, true matches found only by that blocker (marginal recall).
- [ ] Tests per blocker: known pairs found; partitioning respected; dedup mode never returns `i1 >= i2`; frequency caps logged; results identical for the same `random_state`.

## Phase 4 — Comparators and scorers

Comparators turn each candidate pair into features; scorers turn features into one 0–1 score. Both are vectorised over the pairs table: gather values with `left[col].to_numpy()[pairs.i1]`, never `iterrows`.

**Comparators** (`compare/`)

- [ ] `string_sim(fn, left="name_norm", right=None, workers=-1)` → `f_<feature>` float32 in \[0, 1\]:
  - `fn` is one of `ratio, partial_ratio, token_sort_ratio, token_set_ratio, WRatio, QRatio, jaro_winkler, levenshtein_norm`, or any callable `(str, str) -> float`.
  - Implementation: `rapidfuzz.process.cpdist(a, b, scorer=..., workers=workers, dtype=np.float32)` (element-wise, no matrix), divided by 100 for fuzz scorers.
  - Null on either side → `pd.NA`, never 0.
- [ ] `exact(column)` → 1, 0 or NA.
- [ ] `numeric_rel_diff(column)` → `abs(a - b) / max(abs(a), abs(b))`, NA when both are 0 or either is null.
- [ ] `designator_check()` → category `ok, review, reject` from the Phase 2 columns:
  - both `designator_nums` non-empty and different → reject;
  - both `vintage_year` present and different → reject;
  - both `designator_series` present and different → reject;
  - `designator_quals` differ → review;
  - nums on one side only → review ("ABC Fund" vs "ABC Fund II");
  - otherwise ok.
- [ ] `legal_form_compat(equivalent=[[LTD, LIMITED]])` → category `same, equivalent, different, missing`.
- [ ] `country_match(column="country")` → 1, 0 or NA; optional `geo_distance_km(lat, lon)`.
- [ ] `token_idf_overlap()` → reuses the token-index IDF from `ctx.artifacts` when present.
- [ ] `grade(feature, bands={"A": 0.95, "B": 0.80})` → letters A (same), B (similar), F (different), Z (missing), D&B style. `match_grade(features=[...])` concatenates them into one string column `f_match_grade`.

**Scorers** (`scoring/`)

- [ ] `feature(name)`: `score = f_<name>`. Used by the legacy path.
- [ ] `weighted(weights={...}, maps={...}, missing="renormalize")`:
  - `score = Σ w·f / Σ w` over features present for that pair (`missing="renormalize"`), or missing treated as 0 (`missing="zero"`).
  - Categorical features need a map, for example `f_designators: {ok: 1.0, review: 0.5, reject: 0.0}`; an unmapped category raises `ConfigurationError`.
- [ ] `grade_table(path)`: CSV `grade_pattern, confidence` (0–10, `*` wildcard per letter, first match wins); `score = confidence / 10`.
- [ ] `fellegi_sunter(levels={...}, prior=None, estimate_u="random_sample", u_sample_size=1_000_000, max_iter=50, tol=1e-4)`:
  - Each feature is cut into levels (for example `f_name: [0.95, 0.85, 0.70]`, categories as-is, NA = its own level contributing 0).
  - u probabilities estimated from random record pairs (not candidates), m probabilities by EM on the candidates, as Splink does.
  - Match weight = log2 prior odds + Σ log2(m/u); `score` = the posterior probability. Store fitted m and u in `ctx.artifacts` and in `PipelineResult.config` for audit.
- [ ] `model(estimator=path_or_object, features=[...])`: any object with `predict_proba`; `score = predict_proba[:, 1]`.
- [ ] `evaluation/train.py` — `train_scorer(pairs, labels, features, model="hist_gbdt", calibrate="isotonic", cv=5, random_state=0)`:
  - `HistGradientBoostingClassifier` (handles NA and categories natively) wrapped in `CalibratedClassifierCV`.
  - Labels can come from LLM decisions (distillation) or from human review; returns the fitted estimator plus a metrics table per score band.
- [ ] Tests: every comparator on nulls, empty strings and unicode; weighted renormalisation; Fellegi–Sunter recovers known m/u on synthetic data within 0.05; grade-table wildcards.

## Phase 5 — LLM layer

The LLM validator becomes four swappable parts: provider × prompt template × output parser × decision policy. A failed or missing answer never becomes a match by default in the new engine.

```python
@dataclass
class ProviderResponse:
    text: str | None
    json: dict | None          # structured output, when the provider returned one
    probabilities: dict | None # label -> probability, for classifier endpoints
    usage: dict                # input_tokens, output_tokens
    raw: Any

class Provider(Protocol):
    name: str
    def complete(self, messages: list[dict], *, schema: dict | None = None, temperature: float = 0.0,
                 max_tokens: int = 4096) -> ProviderResponse: ...
```

**Providers** (`llm/providers/`)

- [ ] `openai(model, api_key=None, base_url=None, structured=True)`: Chat Completions; when `structured` and a schema is given, `response_format={"type": "json_schema", ...}`.
- [ ] `anthropic(model, api_key=None, max_tokens=4096, structured=True)`: Messages API; structured output via one forced tool (`tool_choice={"type": "tool", "name": "record_decisions"}`, the schema as `input_schema`). Fixes the hard-coded `max_tokens=1000` (validators.py L489).
- [ ] `databricks(endpoint, host=None, token=None, mode="chat")`:
  - `mode="chat"`: OpenAI-compatible client on `{host}/serving-endpoints`, as today.
  - `mode="invocations"`: POST `{host}/serving-endpoints/{endpoint}/invocations` with a payload built from `request_template` (Jinja, JSON) and the answer read from `response_path` (for example `predictions[0].probabilities`). This covers custom classifier endpoints that return per-option probabilities, such as `semif_llama`.
  - Host and token from `DATABRICKS_HOST`, `DATABRICKS_TOKEN` or `DATABRICKS_API_KEY`, as today.
- [ ] `callable(fn)` for custom code and `fake(responses)` for tests (cycles through canned `ProviderResponse`s and records every prompt).
- [ ] Shared in `providers/base.py`:
  - Retries on 429, 5xx and timeouts: exponential backoff with full jitter, base 1 s, cap 30 s, `max_retries=3`.
  - `timeout=60` s.
  - Optional rate limit through the existing `security/rate_limiter.py` (`requests_per_minute`).
  - Token usage summed into `PipelineResult.stats`.

**Prompt templates** (`llm/templates.py`, `llm/prompts/`)

- [ ] Jinja2 with `StrictUndefined`; a template is a package file name or a user path.
- [ ] Variables: `pairs` (list of `{idx, left: {name, <metadata>}, right: {...}, features: {...}}`), `instructions`, `schema`.
- [ ] `pair_validation_v1.j2`: reproduces the legacy wording of `_create_prompt` (validators.py L401–441) byte for byte, including the similarity score. Used only by the legacy class; locked by the Phase 0 snapshot.
- [ ] `pair_validation_v2.j2`: no similarity score (it anchors the model); shows legal form, designators and any `include_features` the user whitelists; asks for `{pair_index, is_duplicate, confidence, reason}` per pair.
- [ ] `pair_index` in prompts is the local position 0..k-1 inside the batch, mapped back to the pairs index after parsing.

**Parsers** (`llm/parsers.py`)

- [ ] `json_decisions`: pydantic model `Decision(pair_index: int, is_duplicate: bool, confidence: float | None, reason: str = "")`. Uses `response.json` when present, else the legacy `_extract_json` fallbacks (fenced block, first `{...}`). A pair missing from the answer is undecided, not a match.
- [ ] `probabilities(positive_label="yes")`: reads `response.probabilities`; for classifier endpoints, one pair per call.
- [ ] Any callable `(ProviderResponse, batch) -> list[Decision]` is a valid parser.

**Decision policy** (`llm/policy.py`)

- [ ] `DecisionPolicy(mode="bool", match_above=0.7, non_match_below=0.3, on_failure="unvalidated", on_missing="unvalidated")`:
  - `mode="bool"` uses `is_duplicate`; `mode="probability"` uses `confidence` or the positive-label probability; between the two thresholds stays undecided.
  - `on_failure` applies when a batch fails after retries; values `unvalidated`, `match`, `non_match`.
  - Every failed pair gets `decision_reason="llm_failed: <error class>"`.

**Validator** (`llm/validator.py`)

- [ ] `LLMValidator(provider, prompt="pair_validation_v2.j2", parser="json_decisions", policy=DecisionPolicy(), batch_size=10, n_workers=4, scope="review", metadata_columns=[], instructions=None, cache="sqlite:///.deduplix_llm_cache.db", checkpoint_every_n_batches=0)`.
- [ ] Metadata lookup through `left.set_index(id)` built once per run (fixes the per-pair DataFrame filter at validators.py L412–413).
- [ ] Cache: SQLite table keyed by sha256 of provider name, model, template name and version, instructions, and the pair's rendered content; stores the decision. Reruns and threshold tuning cost nothing for pairs already decided.
- [ ] `LLMConsolidationEngine` (llm\_consolidation.py) switched to the same providers, templates and parsers internally; `LLMConsolidationConfig` keeps its fields.
- [ ] Tests with `FakeProvider`: partial answers, malformed JSON, a failing batch under each `on_failure` value, probability mode thresholds, cache hits, v1 template snapshot.

## Phase 6 — Clustering and consolidation

Connected components stays the legacy default, but one bad edge can merge two groups (A\~B, B\~C ⇒ A=C), so the engine adds size guards and a community-detection option.

- [ ] `components` — `ConnectedComponents()`: `scipy.sparse.csgraph.connected_components` on a CSR adjacency built from match edges; falls back to networkx when scipy is missing. Replaces the `iterrows` loops in `_cluster_entities` (core.py L413, L425).
- [ ] `guarded` — `GuardedComponents(max_size=50, strategy="drop_weakest", min_edge_score=None)`:
  - `drop_weakest`: inside an oversized component, remove the lowest-score edge, recompute, repeat until every component is ≤ `max_size`.
  - `louvain`: split oversized components with `networkx.community.louvain_communities(weight="score", resolution=1.0, seed=random_state)`.
  - `flag`: keep the component and set `cluster_flag="oversized"`.
  - Removed edges are written back as `decision="non_match"`, `decided_by="cluster.guarded"`, so the pairs table stays the single source of truth.
- [ ] `cluster/report.py` — `cluster_report(pairs, clusters)`: per cluster size, match-edge density, minimum internal score; flags clusters with density below 0.5 for review.
- [ ] Legacy group ids in `to_legacy()`: singletons get 0; groups numbered from 1 in order of their smallest original row position, so numbering is deterministic.
- [ ] Consolidation (link mode): register `ConsolidationEngine` as consolidator `rules` and `LLMConsolidationEngine` as `llm`, both fed from `to_legacy_cross()`. Their config classes and output columns stay as they are.
- [ ] Tests: a chain A-B-C-D with one weak edge splits under `drop_weakest`; Louvain is reproducible with a fixed seed; legacy numbering matches the Phase 0 golden files.

## Phase 7 — Backward-compatibility wiring

Legacy classes keep their signatures and outputs; they gain appended keyword arguments that give access to the engine. Phase 7 is done when the Phase 0 golden and API-snapshot tests pass unchanged and the README examples run.

- [ ] `FuzzyMatcher`:
  - Same constructor; new trailing kwarg `blocking=None`. With `None`, `find_matches` and `find_cross_matches` run the exact `cdist` code path (now `BruteForceBlocker`), so the golden files match.
  - With `blocking=[...]` (blocker specs), candidates come from those blockers and are scored with `cpdist` using the same scorer, threshold and preprocessing; output columns unchanged.
  - `n_workers` default: see open decisions (tests and README say 4, code says 1).
- [ ] `EngineMatcher(pipeline="pipeline.yaml" | dict | Pipeline)`: implements `find_matches` and `find_cross_matches` by running normalize → block → compare → score and returning a legacy `MatchResult`. One-line speed-up for existing code: `DeduplicationPipeline(matcher=EngineMatcher("pipeline.yaml"))`.
- [ ] `DeduplicationPipeline.__init__(matcher, validator=None, checkpoint=True, checkpoint_dir=".deduplix_checkpoints", checkpoint_type=_UNSET, checkpoint_db_path=None, checkpoint_compress=False, hooks=None)`:
  - `checkpoint_type` not passed → legacy `core.Checkpointer` (so existing checkpoint folders still resume) and `self.checkpoint_type = "file"`.
  - `"file"` → `FileCheckpointer(checkpoint_dir, compress=checkpoint_compress)`; `"database"` → `DatabaseCheckpointer(checkpoint_db_path or "deduplix_checkpoints.db", compress=checkpoint_compress)`; anything else → `ConfigurationError("Unsupported checkpoint type: <value>")`.
- [ ] `run()` raises `DataValidationError` for missing columns. Make `DataValidationError` and `ConfigurationError` inherit from `(DeduplixError, ValueError)` so callers catching `ValueError` keep working.
- [ ] Validator adapters in both directions:
  - `as_engine_validator(legacy)`: calls `legacy.validate(MatchResult(pairs), original_df=...)`; validated rows → `match`, removed rows → `non_match`, `validation_reason` → `decision_reason`.
  - `as_legacy_validator(engine_validator)`: exposes `.validate(match_result, original_df, **kwargs)` returning `ValidationResult`.
  - `DeduplicationPipeline` accepts either kind and adapts automatically.
- [ ] Legacy `LLMValidator`: same 13 parameters; new trailing kwargs `on_failure="match"` (today's behaviour), `prompt_template="pair_validation_v1.j2"`, `parser=None`, `max_tokens=4096`, `cache=None`. Built internally from the Phase 5 parts. When a fallback fires, emit one `DeduplixWarning` per run and record `failed_batches` and `failed_pairs` in the result metadata.
- [ ] `RuleBasedValidator`: same behaviour, vectorised (pre-indexed metadata lookups instead of per-row filters at validators.py L66–117, no `iterrows` at L158). Golden tests must still match, including `validation_reason` text.
- [ ] `deduplix/__init__.py`: keep all 10 exports; add `Pipeline`, `PipelineResult`, `register`, `load_pipeline`, `EngineMatcher`. Bump `__version__` to `0.2.0`.
- [ ] `utils.load_config` unchanged; add `utils.load_pipeline` re-export.

## Phase 8 — Evaluation and benchmarks

No threshold, blocker or scorer becomes a default without numbers from this phase. Gold labels are a DataFrame `id1, id2, label` (1 = same entity).

- [ ] `evaluation/metrics.py`:
  - `pair_metrics(pred, gold)`: precision, recall, F1 at pair level.
  - `metrics_by_band(pairs, gold, bands=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95])`: precision per score band, to set router thresholds.
  - `cluster_metrics(clusters, gold_clusters)`: B-cubed precision and recall, pairwise F1.
  - Re-export `blocking_report` from Phase 3.
- [ ] `evaluation/synthetic.py` — `make_synthetic(n_entities, dup_rate=0.3, variants=[...], seed=42)` returning records and gold. Variants: typo, legal-form swap, abbreviation, word order, accent removal, transliteration, and hard negatives: fund-number neighbours ("Fund I" vs "Fund IV"), feeder vs master, manager vs fund, same name in another country.
- [ ] `evaluation/silver.py` — silver standard on real data:
  1. Sample `n_seed=5000` left records, stratified by country.
  2. Brute-force them against all of right with `token_set_ratio >= 60`.
  3. Label with an LLM validator (any provider) and export `review.csv` (`id1, id2, llm_label, reason, human_label`).
  4. `import_review(csv)` merges human labels; human wins over LLM. Save `gold.parquet`.
- [ ] `benchmarks/run.py` (marked `slow`, never in CI by default):
  - Datasets: synthetic 10k, 165k (the README's figure), 1M.
  - Configurations: legacy `cdist`; TF-IDF only; TF-IDF + token index + exact keys; plus embeddings.
  - Reported per configuration: seconds per stage, peak RSS (psutil), candidate pairs, reduction ratio, pair completeness, final precision, recall, F1, LLM calls and tokens.
  - Output `benchmarks/results/<date>.md` and `.json`.
- [ ] Acceptance criteria for making the blocked pipeline the recommended default (to be checked, not assumed):
  - Pair completeness ≥ 0.99 against brute force at the same threshold on synthetic 165k.
  - End-to-end at least 10× faster than legacy on synthetic 165k.
  - F1 within 0.5 points of legacy, or better.
  - Zero fund-number hard negatives accepted when `designator_check` is configured.

## Phase 9 — Bug fixes and housekeeping

Each fix is a separate commit with a test, so it can be reviewed and reverted on its own. Line numbers refer to `master` at 0.1.0.

| File and line | Problem | Fix | Changes output? |
| --- | --- | --- | --- |
| validators.py L518, L527, L639 | Failed batch, missing field and missing decision all become `is_duplicate=True` | Policy from Phase 5; legacy default unchanged plus warning and counts | No (legacy), yes in new engine |
| validators.py L489 | Anthropic `max_tokens=1000` truncates batches with reasons | Parameter, default 4096 | Fewer failed batches |
| validators.py L412–413; L66–117 | DataFrame filter per pair for metadata | Index by id once per call | No |
| matchers.py L342; core.py L413, L425, L714; validators.py L158 | `iterrows` loops | Vectorise | No |
| core.py L96–115 | `remove_duplicates` filters the whole frame per entity | Map id → first and last row position once | No |
| core.py L390 | Data hash ignores the name column, so edited data resumes stale checkpoints | Hash id and name columns | Old checkpoints do not resume once |
| core.py L10–14 | Silent `ImportError` fallback for utils hides real errors | Remove the try/except | No |
| matchers.py L83 | `n_workers` default 1; docstring says auto, README and tests say 4 | Per open decision | Speed only |
| matchers.py `find_matches` | `additional_columns` accepted and ignored | Document as ignored; route to comparators in `EngineMatcher` | No |
| `deduplix/consolidation (1).py` | Identical copy of consolidation.py, imported nowhere | Delete | No |
| validation/schemas.py | pydantic v1 `validator`/`root_validator` | `field_validator` and `model_validator(mode="after")` | No |
| setup.py | `pydantic>=1.10` (README says 2); `llm` extra pins `openai>=0.27` (code needs 1.x); no anthropic extra | Update pins and extras per Phase 1 table | No |
| setup.py, README | Distribution `deduplix` vs README `pip install fuzzy-ai`; console script `deduplix` vs README `fuzzy-ai`; URLs `worldbank/fuzzy-ai` vs repo `worldbank/FuzzyAI` | Per open decision | No |
| README vs cli.py | README YAML has `consolidation`, `llm_consolidation`, `security` sections the CLI never reads | CLI `--pipeline` path reads the full config | No |

- [ ] Keep `deduplix/validators.py` (pair validators) and `deduplix/validation/` (input checks) at their current paths; add module docstrings saying which is which instead of renaming.

## Phase 10 — CLI, docs, packaging, release

The CLI gains engine commands without changing the existing ones; the docs are generated from the registry and the config models so they cannot drift.

- [ ] CLI additions (cli.py):
  - `deduplix run --pipeline pipeline.yaml -i data.csv -o out/`. Without `--pipeline`, `run` behaves exactly as today.
  - `deduplix link --pipeline pipeline.yaml --left a.csv --right b.csv -o out/`.
  - `deduplix components [--kind blocker]` lists registered components with parameters.
  - `deduplix validate-config pipeline.yaml` builds the pipeline without data and reports errors.
  - `deduplix explain out/ ID1 ID2` prints every column for one pair.
  - `deduplix evaluate out/ --gold gold.csv` prints pair, band and cluster metrics.
  - `deduplix init --template basic | funds | llm` writes a starter `pipeline.yaml`.
  - Add `fuzzy-ai` as a second console-script name pointing at the same CLI.
- [ ] Ship `deduplix/templates/funds.yaml`, the reference config for fund vehicles:

```yaml
normalize:
  - default_text
  - name_parser: {country_column: country}
blocking:
  - exact: {keys: [lei, registration_no]}
  - tfidf_char: {on: name_base, top_n: 20, threshold: 0.5, partition_by: [country]}
  - token_index: {on: name_base, max_df: 1000, top_k: 50}
compare:
  name: {fn: token_set_ratio, left: name_base}
  designators: {fn: designator_check}
  legal_form: {fn: legal_form_compat}
  country: {fn: exact, left: country}
score:
  weighted:
    weights: {f_name: 0.7, f_country: 0.2, f_legal_form: 0.1}
    maps: {f_legal_form: {same: 1.0, equivalent: 1.0, missing: 0.5, different: 0.0}}
route:
  rule:
    rules:
      - {feature: f_designators, equals: reject, route: reject}
      - {feature: f_designators, equals: review, route: review}
    then: {threshold: {accept_above: 0.95, reject_below: 0.60}}
validate:
  - llm:
      provider: {databricks: {endpoint: "${DEDUPLIX_ENDPOINT}", mode: chat}}
      prompt: pair_validation_v2.j2
      policy: {mode: bool, on_failure: unvalidated}
      metadata_columns: [country, legal_form, designator_nums, designator_quals]
cluster:
  guarded: {max_size: 50, strategy: drop_weakest}
```

- [ ] Docs with mkdocs-material in `docs/`:
  - Concepts: stages, the pairs table, routing.
  - Config reference generated from the pydantic models' JSON schema.
  - Component catalogue generated from `list_components()` and docstrings.
  - "Write your own component" guide: a function, a class, an entry point.
  - Migration guide 0.1 → 0.2: nothing required; how to opt in step by step (`EngineMatcher`, then the full `Pipeline`).
  - Cookbook: fund vehicles, cross-dataset with LEI, LLM through a Databricks endpoint, distilling LLM labels into a GBDT scorer.
- [ ] README: fix install name and URLs; add a 10-line engine quick start; keep every existing example working.
- [ ] `CHANGELOG.md` for 0.2.0: Added, Fixed (Phase 9 table), Behaviour changes (only those marked yes in Phase 9), nothing Removed.
- [ ] Release: version 0.2.0 in `__init__.py` and setup.py, tag `v0.2.0`, build sdist and wheel, install-test the wheel in a clean environment with core dependencies only.

## Open decisions

Each has a recommendation; settle the first four before Phase 1 starts, because they change signatures or packaging.

| # | Decision | Options | Recommendation |
| --- | --- | --- | --- |
| 1 | Minimum Python | 3.8 (setup.py today) or 3.9 | 3.9: 3.8 is end-of-life since October 2024, and current pandas and scikit-learn no longer support it |
| 2 | Legacy `LLMValidator` on failure | keep "match" or switch to "unvalidated" | Keep "match" with a warning in 0.2; switch in 0.3 after one release of `FutureWarning` |
| 3 | Distribution and CLI name | `deduplix` or `fuzzy-ai` | Publish as `deduplix`, add `fuzzy-ai` as a CLI alias, fix the README |
| 4 | `FuzzyMatcher.n_workers` default | 1 (code) or 4 (README, tests) | 4: results do not change, only speed |
| 5 | Data hash includes the name column | yes or no | Yes; existing checkpoints stop resuming once, but stale resumes are a correctness bug |
| 6 | TF-IDF blocker as a core dependency | core or `[blocking]` extra | Extra; `Pipeline.default()` uses it when installed and falls back to brute force with a warning |
| 7 | Default embedding encoder | multilingual-e5-small locally or a Databricks embeddings endpoint | Local e5-small for portability; Databricks encoder documented for IFC use |
| 8 | Router thresholds in `Pipeline.default()` | fixed or required | Fixed at 0.95 / 0.60 for the default only; templates set their own; revisit after Phase 8 numbers |
| 9 | rapidfuzz minimum version | keep ≥2.0 or raise to ≥3.6 | Raise to 3.6 for `cpdist`; the legacy `cdist` path works on it unchanged |
