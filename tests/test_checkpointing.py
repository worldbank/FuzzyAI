"""
Tests for checkpointing functionality (file and database).
"""

import pytest
import pandas as pd
import sqlite3
import json
from pathlib import Path

from deduplix.checkpointing import (
    BaseCheckpointer, FileCheckpointer, DatabaseCheckpointer
)
from deduplix.exceptions import CheckpointError


class TestBaseCheckpointer:
    """Test abstract base checkpointer"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseCheckpointer cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseCheckpointer()

    def test_validation_methods_exist(self):
        """Test that validation methods are available"""
        # These should be accessible as static methods
        BaseCheckpointer.validate_stage_name("test_stage")
        BaseCheckpointer.validate_data_hash("abc123def")

    def test_validate_stage_name_valid(self):
        """Test stage name validation with valid names"""
        valid_names = ["matching", "validation", "test_stage", "stage_1"]

        for name in valid_names:
            BaseCheckpointer.validate_stage_name(name)  # Should not raise

    def test_validate_stage_name_invalid(self):
        """Test stage name validation with invalid names"""
        invalid_names = ["", "  ", "stage with spaces", "stage/with/slashes", None]

        for name in invalid_names:
            with pytest.raises(ValueError):
                BaseCheckpointer.validate_stage_name(name)

    def test_validate_data_hash_valid(self):
        """Test data hash validation with valid hashes"""
        valid_hashes = ["abc123", "1234567890abcdef", "short_hash"]

        for hash_val in valid_hashes:
            BaseCheckpointer.validate_data_hash(hash_val)  # Should not raise

    def test_validate_data_hash_invalid(self):
        """Test data hash validation with invalid hashes"""
        invalid_hashes = ["", "  ", None, "hash with spaces"]

        for hash_val in invalid_hashes:
            with pytest.raises(ValueError):
                BaseCheckpointer.validate_data_hash(hash_val)


class TestFileCheckpointer:
    """Test file-based checkpointing"""

    def test_checkpointer_creation(self, temp_dir):
        """Test basic file checkpointer creation"""
        checkpointer = FileCheckpointer(checkpoint_dir=str(temp_dir))

        assert checkpointer.checkpoint_dir.exists()
        assert checkpointer.compress is True  # Default

    def test_checkpointer_custom_settings(self, temp_dir):
        """Test file checkpointer with custom settings"""
        checkpointer = FileCheckpointer(
            checkpoint_dir=str(temp_dir),
            compress=False
        )

        assert checkpointer.compress is False

    def test_save_and_load_basic(self, file_checkpointer):
        """Test basic save and load functionality"""
        # Create test data
        test_data = pd.DataFrame({
            'id1': ['A', 'B'],
            'id2': ['C', 'D'],
            'similarity_score': [95.0, 88.0]
        })

        stage = "test_stage"
        data_hash = "test_hash"

        # Save
        file_checkpointer.save(test_data, stage, data_hash)

        # Load
        loaded_data = file_checkpointer.load(stage, data_hash)

        # Verify
        assert loaded_data is not None
        assert len(loaded_data) == len(test_data)
        assert list(loaded_data.columns) == list(test_data.columns)
        pd.testing.assert_frame_equal(loaded_data, test_data)

    def test_load_nonexistent(self, file_checkpointer):
        """Test loading non-existent checkpoint"""
        result = file_checkpointer.load("nonexistent", "hash")
        assert result is None

    def test_exists_functionality(self, file_checkpointer):
        """Test checkpoint existence checking"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})
        stage = "test_stage"
        data_hash = "test_hash"

        # Should not exist initially
        assert not file_checkpointer.exists(stage, data_hash)

        # Save and check existence
        file_checkpointer.save(test_data, stage, data_hash)
        assert file_checkpointer.exists(stage, data_hash)

    def test_delete_checkpoint(self, file_checkpointer):
        """Test deleting checkpoints"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})
        stage = "test_stage"
        data_hash = "test_hash"

        # Save, verify exists, delete, verify gone
        file_checkpointer.save(test_data, stage, data_hash)
        assert file_checkpointer.exists(stage, data_hash)

        deleted = file_checkpointer.delete(stage, data_hash)
        assert deleted is True
        assert not file_checkpointer.exists(stage, data_hash)

    def test_clear_all_checkpoints(self, file_checkpointer):
        """Test clearing all checkpoints"""
        # Save multiple checkpoints
        test_data = pd.DataFrame({'col': [1, 2, 3]})

        for i in range(3):
            file_checkpointer.save(test_data, f"stage_{i}", f"hash_{i}")

        # Clear all
        cleared_count = file_checkpointer.clear()
        assert cleared_count >= 3  # Might include metadata files

        # Verify all gone
        for i in range(3):
            assert not file_checkpointer.exists(f"stage_{i}", f"hash_{i}")

    def test_list_checkpoints(self, file_checkpointer):
        """Test listing checkpoints"""
        # Save some checkpoints
        test_data = pd.DataFrame({'col': [1, 2, 3]})

        file_checkpointer.save(test_data, "stage1", "hash1")
        file_checkpointer.save(test_data, "stage2", "hash1")  # Same hash, different stage

        # List all
        all_checkpoints = file_checkpointer.list_checkpoints()
        assert len(all_checkpoints) >= 2

        # List for specific hash
        hash1_checkpoints = file_checkpointer.list_checkpoints(data_hash="hash1")
        assert len(hash1_checkpoints) == 2

    def test_get_metadata(self, file_checkpointer):
        """Test getting checkpoint metadata"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})
        metadata = {'custom': 'value'}

        file_checkpointer.save(test_data, "stage", "hash", metadata=metadata)

        retrieved_metadata = file_checkpointer.get_metadata("stage", "hash")
        assert retrieved_metadata is not None
        assert retrieved_metadata['custom_metadata']['custom'] == 'value'
        assert 'stage' in retrieved_metadata
        assert 'rows' in retrieved_metadata

    def test_compression_functionality(self, temp_dir):
        """Test compression on/off"""
        # Test with compression
        compressed_checkpointer = FileCheckpointer(
            checkpoint_dir=str(temp_dir / "compressed"),
            compress=True
        )

        # Test without compression
        uncompressed_checkpointer = FileCheckpointer(
            checkpoint_dir=str(temp_dir / "uncompressed"),
            compress=False
        )

        # Same data
        test_data = pd.DataFrame({'data': range(1000)})  # Larger data for compression effect

        compressed_checkpointer.save(test_data, "stage", "hash")
        uncompressed_checkpointer.save(test_data, "stage", "hash")

        # Both should load correctly
        compressed_loaded = compressed_checkpointer.load("stage", "hash")
        uncompressed_loaded = uncompressed_checkpointer.load("stage", "hash")

        pd.testing.assert_frame_equal(compressed_loaded, test_data)
        pd.testing.assert_frame_equal(uncompressed_loaded, test_data)


class TestDatabaseCheckpointer:
    """Test database-based checkpointing"""

    def test_checkpointer_creation(self, temp_db_path):
        """Test database checkpointer creation"""
        checkpointer = DatabaseCheckpointer(db_path=temp_db_path)

        assert checkpointer.db_path == temp_db_path
        assert checkpointer.compress is True  # Default

        # Database should be created
        assert Path(temp_db_path).exists()

    def test_database_schema_creation(self, temp_db_path):
        """Test that database schema is created correctly"""
        checkpointer = DatabaseCheckpointer(db_path=temp_db_path)

        # Check schema
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='checkpoints'")
            schema = cursor.fetchone()

            assert schema is not None
            assert 'stage' in schema[0]
            assert 'data_hash' in schema[0]
            assert 'data_bytes' in schema[0]

    def test_save_and_load_basic(self, db_checkpointer):
        """Test basic database save and load"""
        test_data = pd.DataFrame({
            'id1': ['A', 'B'],
            'id2': ['C', 'D'],
            'similarity_score': [95.0, 88.0]
        })

        stage = "test_stage"
        data_hash = "test_hash"

        # Save
        db_checkpointer.save(test_data, stage, data_hash)

        # Load
        loaded_data = db_checkpointer.load(stage, data_hash)

        # Verify
        assert loaded_data is not None
        assert len(loaded_data) == len(test_data)
        pd.testing.assert_frame_equal(loaded_data, test_data)

    def test_save_with_metadata(self, db_checkpointer):
        """Test saving with custom metadata"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})
        metadata = {'custom_key': 'custom_value', 'number': 42}

        db_checkpointer.save(test_data, "stage", "hash", metadata=metadata)

        retrieved_metadata = db_checkpointer.get_metadata("stage", "hash")
        assert retrieved_metadata is not None
        assert retrieved_metadata['custom_metadata']['custom_key'] == 'custom_value'
        assert retrieved_metadata['custom_metadata']['number'] == 42

    def test_exists_functionality(self, db_checkpointer):
        """Test checkpoint existence checking"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})

        # Should not exist initially
        assert not db_checkpointer.exists("stage", "hash")

        # Save and check
        db_checkpointer.save(test_data, "stage", "hash")
        assert db_checkpointer.exists("stage", "hash")

    def test_delete_checkpoint(self, db_checkpointer):
        """Test deleting checkpoints"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})

        # Save, verify, delete, verify
        db_checkpointer.save(test_data, "stage", "hash")
        assert db_checkpointer.exists("stage", "hash")

        deleted = db_checkpointer.delete("stage", "hash")
        assert deleted is True
        assert not db_checkpointer.exists("stage", "hash")

    def test_list_checkpoints(self, db_checkpointer):
        """Test listing checkpoints"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})

        # Save multiple checkpoints
        db_checkpointer.save(test_data, "stage1", "hash1")
        db_checkpointer.save(test_data, "stage2", "hash1")  # Same hash
        db_checkpointer.save(test_data, "stage1", "hash2")  # Same stage

        # List all
        all_checkpoints = db_checkpointer.list_checkpoints()
        assert len(all_checkpoints) == 3

        # List for specific hash
        hash1_checkpoints = db_checkpointer.list_checkpoints(data_hash="hash1")
        assert len(hash1_checkpoints) == 2

    def test_clear_functionality(self, db_checkpointer):
        """Test clearing checkpoints"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})

        # Save some checkpoints
        for i in range(3):
            db_checkpointer.save(test_data, f"stage_{i}", "common_hash")

        # Clear specific hash
        cleared = db_checkpointer.clear(data_hash="common_hash")
        assert cleared == 3

        # Verify cleared
        remaining = db_checkpointer.list_checkpoints(data_hash="common_hash")
        assert len(remaining) == 0

    def test_get_statistics(self, db_checkpointer):
        """Test getting database statistics"""
        test_data = pd.DataFrame({'col': range(100)})  # Larger dataset

        # Save some checkpoints
        db_checkpointer.save(test_data, "stage1", "hash1")
        db_checkpointer.save(test_data, "stage2", "hash2")

        stats = db_checkpointer.get_statistics()

        # Check expected statistics
        assert 'total_checkpoints' in stats
        assert 'total_size_bytes' in stats
        assert 'unique_stages' in stats
        assert 'unique_datasets' in stats
        assert stats['total_checkpoints'] >= 2
        assert stats['unique_stages'] >= 2
        assert stats['unique_datasets'] >= 2

    def test_cleanup_old_checkpoints(self, db_checkpointer):
        """Test cleaning up old checkpoints"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})

        # Save a checkpoint (will have recent timestamp)
        db_checkpointer.save(test_data, "recent_stage", "hash1")

        # Cleanup checkpoints older than 1 day (should not affect recent one)
        cleaned = db_checkpointer.cleanup_old_checkpoints(days_old=1)

        # Recent checkpoint should still exist
        assert db_checkpointer.exists("recent_stage", "hash1")

    def test_optimize_database(self, db_checkpointer):
        """Test database optimization"""
        # Should not raise any exceptions
        db_checkpointer.optimize_database()

    def test_export_checkpoints(self, db_checkpointer, temp_dir):
        """Test exporting checkpoints to files"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})
        metadata = {'test': 'metadata'}

        # Save some checkpoints
        db_checkpointer.save(test_data, "stage1", "hash1", metadata=metadata)
        db_checkpointer.save(test_data, "stage2", "hash1", metadata=metadata)

        export_path = temp_dir / "exports"

        # Export all checkpoints for hash1
        exported_count = db_checkpointer.export_checkpoints(str(export_path), data_hash="hash1")

        assert exported_count == 2

        # Check exported files exist
        assert (export_path / "stage1_hash1.parquet").exists()
        assert (export_path / "stage1_hash1.meta.json").exists()
        assert (export_path / "stage2_hash1.parquet").exists()

    def test_compression_setting(self, temp_dir):
        """Test compression enabled/disabled"""
        db_path_compressed = str(temp_dir / "compressed.db")
        db_path_uncompressed = str(temp_dir / "uncompressed.db")

        compressed_checkpointer = DatabaseCheckpointer(
            db_path=db_path_compressed, compress=True
        )
        uncompressed_checkpointer = DatabaseCheckpointer(
            db_path=db_path_uncompressed, compress=False
        )

        # Same large dataset
        test_data = pd.DataFrame({'data': range(1000)})

        compressed_checkpointer.save(test_data, "stage", "hash")
        uncompressed_checkpointer.save(test_data, "stage", "hash")

        # Both should load correctly
        compressed_loaded = compressed_checkpointer.load("stage", "hash")
        uncompressed_loaded = uncompressed_checkpointer.load("stage", "hash")

        pd.testing.assert_frame_equal(compressed_loaded, test_data)
        pd.testing.assert_frame_equal(uncompressed_loaded, test_data)

    def test_concurrent_access(self, db_checkpointer):
        """Test basic concurrent access safety"""
        import threading

        test_data = pd.DataFrame({'col': range(10)})
        results = []

        def save_checkpoint(stage_suffix):
            try:
                db_checkpointer.save(test_data, f"stage_{stage_suffix}", f"hash_{stage_suffix}")
                results.append(True)
            except Exception:
                results.append(False)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=save_checkpoint, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should succeed
        assert all(results)
        assert len(results) == 5


class TestCheckpointingErrorHandling:
    """Test error handling in checkpointing"""

    def test_file_checkpointer_invalid_directory(self):
        """Test file checkpointer with invalid directory"""
        # Try to create in a read-only location (this might vary by OS)
        with pytest.raises(CheckpointError):
            FileCheckpointer(checkpoint_dir="/invalid/readonly/path")

    def test_database_checkpointer_invalid_path(self):
        """Test database checkpointer with invalid path"""
        with pytest.raises(CheckpointError):
            DatabaseCheckpointer(db_path="/invalid/readonly/path/test.db")

    def test_save_with_invalid_stage_name(self, file_checkpointer):
        """Test saving with invalid stage name"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})

        with pytest.raises(CheckpointError) as exc_info:
            file_checkpointer.save(test_data, "", "valid_hash")  # Empty stage name

        assert "stage" in str(exc_info.value).lower()

    def test_save_with_invalid_data_hash(self, file_checkpointer):
        """Test saving with invalid data hash"""
        test_data = pd.DataFrame({'col': [1, 2, 3]})

        with pytest.raises(CheckpointError) as exc_info:
            file_checkpointer.save(test_data, "valid_stage", "")  # Empty hash

        assert "hash" in str(exc_info.value).lower()

    def test_load_corrupted_data(self, file_checkpointer, temp_dir):
        """Test loading corrupted checkpoint data"""
        # Create a corrupted parquet file
        corrupted_path = file_checkpointer.checkpoint_dir / "corrupted_stage_hash.parquet"
        with open(corrupted_path, 'w') as f:
            f.write("This is not parquet data")

        # Should return None for corrupted data (graceful failure)
        result = file_checkpointer.load("corrupted_stage", "hash")
        assert result is None