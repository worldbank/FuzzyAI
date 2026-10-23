# Baseline test results (master, 0.1.0)

Environment: Python 3.13, pandas 3.1, rapidfuzz 3.14, pydantic 2.14, Windows 11.

`pytest` before any change: **64 failed, 64 passed**, after adding the missing `performance` marker to `pyproject.toml` (without it `tests/test_matchers.py` fails at collection).

## Failures by file

### test_checkpointing.py (26)

- `TestBaseCheckpointer::test_validation_methods_exist`
- `TestBaseCheckpointer::test_validate_stage_name_valid`
- `TestBaseCheckpointer::test_validate_stage_name_invalid`
- `TestBaseCheckpointer::test_validate_data_hash_valid`
- `TestBaseCheckpointer::test_validate_data_hash_invalid`
- `TestFileCheckpointer::test_checkpointer_creation`
- `TestFileCheckpointer::test_checkpointer_custom_settings`
- `TestFileCheckpointer::test_save_and_load_basic`
- `TestFileCheckpointer::test_exists_functionality`
- `TestFileCheckpointer::test_delete_checkpoint`
- `TestFileCheckpointer::test_clear_all_checkpoints`
- `TestFileCheckpointer::test_list_checkpoints`
- `TestFileCheckpointer::test_get_metadata`
- `TestFileCheckpointer::test_compression_functionality`
- `TestDatabaseCheckpointer::test_save_and_load_basic`
- `TestDatabaseCheckpointer::test_save_with_metadata`
- `TestDatabaseCheckpointer::test_exists_functionality`
- `TestDatabaseCheckpointer::test_delete_checkpoint`
- `TestDatabaseCheckpointer::test_list_checkpoints`
- `TestDatabaseCheckpointer::test_clear_functionality`
- `TestDatabaseCheckpointer::test_get_statistics`
- `TestDatabaseCheckpointer::test_cleanup_old_checkpoints`
- `TestDatabaseCheckpointer::test_export_checkpoints`
- `TestDatabaseCheckpointer::test_compression_setting`
- `TestDatabaseCheckpointer::test_concurrent_access`
- `TestCheckpointingErrorHandling::test_file_checkpointer_invalid_directory`

### test_core.py (6)

- `TestDeduplicationResult::test_get_group`
- `TestDeduplicationPipeline::test_pipeline_with_file_checkpointing`
- `TestDeduplicationPipeline::test_pipeline_with_database_checkpointing`
- `TestDeduplicationPipeline::test_pipeline_invalid_checkpoint_type`
- `TestDeduplicationPipeline::test_compute_data_hash`
- `TestDeduplicationPipeline::test_run_with_validation_errors`

### test_matchers.py (5)

- `TestFuzzyMatcher::test_matcher_creation_default`
- `TestFuzzyMatcher::test_matcher_creation_custom`
- `TestFuzzyMatcher::test_max_matches_per_entity_limit`
- `TestFuzzyMatcher::test_matching_with_special_characters`
- `TestFuzzyMatcher::test_error_handling_missing_columns`

### test_security.py (18)

- `TestRateLimitConfig::test_invalid_config`
- `TestRateLimiter::test_rate_limit_reset`
- `TestRateLimiter::test_different_clients_separate_limits`
- `TestRateLimiter::test_get_remaining_requests`
- `TestSecuritySanitizer::test_sanitizer_creation`
- `TestSecuritySanitizer::test_sanitize_text_html_removal`
- `TestSecuritySanitizer::test_sanitize_text_length_limit`
- `TestSecuritySanitizer::test_sanitize_text_sql_injection_patterns`
- `TestSecuritySanitizer::test_validate_api_key_invalid_characters`
- `TestSecuritySanitizer::test_detect_malicious_patterns_true_positives`
- `TestResourceMonitor::test_monitor_creation`
- `TestResourceMonitor::test_custom_limits`
- `TestResourceMonitor::test_check_memory_usage`
- `TestResourceMonitor::test_check_thread_count`
- `TestResourceMonitor::test_check_processing_time`
- `TestResourceMonitor::test_get_resource_stats`
- `TestResourceMonitor::test_monitor_context_manager`
- `TestResourceMonitor::test_monitor_with_actual_workload`

### test_validation.py (9)

- `TestEntityDataValidation::test_name_length_validation`
- `TestDeduplicationInputValidation::test_validate_missing_id_column`
- `TestDeduplicationInputValidation::test_validate_missing_name_column`
- `TestDeduplicationInputValidation::test_validate_empty_dataframe`
- `TestDeduplicationInputValidation::test_validate_duplicate_ids`
- `TestDeduplicationInputValidation::test_validate_missing_additional_columns`
- `TestCrossDatasetValidation::test_validate_cross_dataset_missing_columns`
- `TestTextSanitization::test_sanitize_very_long_text`
- `TestTextSanitization::test_sanitize_with_strict_mode`

## Triage

| Group | Cause | Plan |
| --- | --- | --- |
| test_core: checkpoint_type, invalid type, validation errors, data hash | gaps listed in todo.md Phase 0 | Phase 7 and 9 |
| test_core `test_get_group` | not in todo.md | investigate |
| test_matchers: default `n_workers` | code default 1, tests and README say 4 | Open decision 4 |
| test_checkpointing | tests call `validate_stage_name` / `validate_data_hash` with a different contract, expect a `compress` attribute, and use non-hex hashes such as `test_hash` | not in todo.md |
| test_security, test_validation | tests written against an API that differs from `security/` and `validation/` | not in todo.md |
