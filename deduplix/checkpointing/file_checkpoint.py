"""
File-based checkpointing implementation.

Stores checkpoints as Parquet files in a specified directory.
This is a refactored version of the original Checkpointer class.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import warnings

from .base import BaseCheckpointer
from ..exceptions import CheckpointError


class FileCheckpointer(BaseCheckpointer):
    """
    File-based checkpointing using Parquet files.

    Stores checkpoint data as Parquet files with accompanying JSON metadata.
    """

    def __init__(self, checkpoint_dir: str = ".deduplix_checkpoints", **kwargs):
        """
        Initialize file-based checkpointer

        Parameters
        ----------
        checkpoint_dir : str
            Directory to store checkpoint files
        **kwargs
            Additional configuration parameters
        """
        super().__init__(kwargs)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def save(
        self,
        data: pd.DataFrame,
        stage: str,
        data_hash: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save checkpoint data to Parquet file

        Parameters
        ----------
        data : pd.DataFrame
            Data to checkpoint
        stage : str
            Processing stage identifier
        data_hash : str
            Unique hash identifying the dataset
        metadata : Optional[Dict[str, Any]]
            Additional metadata to store
        """
        try:
            checkpoint_id = self.create_checkpoint_id(stage, data_hash)
            data_path = self.checkpoint_dir / f"{checkpoint_id}.parquet"
            meta_path = self.checkpoint_dir / f"{checkpoint_id}.meta.json"

            # Save data as Parquet
            data.to_parquet(data_path)

            # Save metadata as JSON
            checkpoint_metadata = {
                'stage': stage,
                'data_hash': data_hash,
                'timestamp': datetime.now().isoformat(),
                'rows': len(data),
                'columns': list(data.columns),
                'data_path': str(data_path),
                'file_size_bytes': data_path.stat().st_size if data_path.exists() else 0,
                'custom_metadata': metadata or {}
            }

            with open(meta_path, 'w') as f:
                json.dump(checkpoint_metadata, f, indent=2)

        except Exception as e:
            raise CheckpointError(
                f"Failed to save checkpoint for stage '{stage}': {e}",
                checkpoint_stage=stage,
                checkpoint_path=str(self.checkpoint_dir),
                context={'data_hash': data_hash, 'data_shape': data.shape}
            ) from e

    def load(self, stage: str, data_hash: str) -> Optional[pd.DataFrame]:
        """
        Load checkpoint data from Parquet file

        Parameters
        ----------
        stage : str
            Processing stage identifier
        data_hash : str
            Unique hash identifying the dataset

        Returns
        -------
        Optional[pd.DataFrame]
            Loaded checkpoint data, or None if not found
        """
        try:
            checkpoint_id = self.create_checkpoint_id(stage, data_hash)
            data_path = self.checkpoint_dir / f"{checkpoint_id}.parquet"

            if data_path.exists():
                return pd.read_parquet(data_path)
            return None

        except Exception as e:
            warnings.warn(f"Failed to load checkpoint for stage '{stage}': {e}", UserWarning)
            return None

    def exists(self, stage: str, data_hash: str) -> bool:
        """
        Check if checkpoint exists

        Parameters
        ----------
        stage : str
            Processing stage identifier
        data_hash : str
            Unique hash identifying the dataset

        Returns
        -------
        bool
            True if checkpoint exists, False otherwise
        """
        try:
            checkpoint_id = self.create_checkpoint_id(stage, data_hash)
            data_path = self.checkpoint_dir / f"{checkpoint_id}.parquet"
            return data_path.exists()
        except Exception:
            return False

    def list_checkpoints(self, data_hash: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available checkpoints

        Parameters
        ----------
        data_hash : Optional[str]
            Filter by specific dataset hash

        Returns
        -------
        List[Dict[str, Any]]
            List of checkpoint information
        """
        checkpoints = []

        try:
            for meta_file in self.checkpoint_dir.glob("*.meta.json"):
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)

                    # Filter by data hash if specified
                    if data_hash and metadata.get('data_hash') != data_hash:
                        continue

                    checkpoints.append(metadata)

                except Exception as e:
                    warnings.warn(f"Failed to read metadata from {meta_file}: {e}", UserWarning)

        except Exception as e:
            warnings.warn(f"Failed to list checkpoints: {e}", UserWarning)

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return checkpoints

    def delete(self, stage: str, data_hash: str) -> bool:
        """
        Delete a specific checkpoint

        Parameters
        ----------
        stage : str
            Processing stage identifier
        data_hash : str
            Unique hash identifying the dataset

        Returns
        -------
        bool
            True if checkpoint was deleted, False if not found
        """
        try:
            checkpoint_id = self.create_checkpoint_id(stage, data_hash)
            data_path = self.checkpoint_dir / f"{checkpoint_id}.parquet"
            meta_path = self.checkpoint_dir / f"{checkpoint_id}.meta.json"

            deleted = False

            if data_path.exists():
                data_path.unlink()
                deleted = True

            if meta_path.exists():
                meta_path.unlink()
                deleted = True

            return deleted

        except Exception as e:
            warnings.warn(f"Failed to delete checkpoint for stage '{stage}': {e}", UserWarning)
            return False

    def clear(self, data_hash: Optional[str] = None) -> int:
        """
        Clear checkpoints

        Parameters
        ----------
        data_hash : Optional[str]
            Clear only checkpoints for specific dataset hash

        Returns
        -------
        int
            Number of checkpoints deleted
        """
        deleted_count = 0

        try:
            if data_hash:
                # Delete specific checkpoints
                checkpoints = self.list_checkpoints(data_hash)
                for checkpoint in checkpoints:
                    stage = checkpoint.get('stage')
                    if stage and self.delete(stage, data_hash):
                        deleted_count += 1
            else:
                # Delete all checkpoints
                for file_path in self.checkpoint_dir.glob("*.parquet"):
                    file_path.unlink()
                    deleted_count += 1

                for file_path in self.checkpoint_dir.glob("*.meta.json"):
                    file_path.unlink()

        except Exception as e:
            warnings.warn(f"Failed to clear checkpoints: {e}", UserWarning)

        return deleted_count

    def get_metadata(self, stage: str, data_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get checkpoint metadata

        Parameters
        ----------
        stage : str
            Processing stage identifier
        data_hash : str
            Unique hash identifying the dataset

        Returns
        -------
        Optional[Dict[str, Any]]
            Checkpoint metadata, or None if not found
        """
        try:
            checkpoint_id = self.create_checkpoint_id(stage, data_hash)
            meta_path = self.checkpoint_dir / f"{checkpoint_id}.meta.json"

            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    return json.load(f)
            return None

        except Exception as e:
            warnings.warn(f"Failed to get metadata for stage '{stage}': {e}", UserWarning)
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get checkpointing system statistics

        Returns
        -------
        Dict[str, Any]
            Statistics about checkpoint usage and storage
        """
        try:
            checkpoints = self.list_checkpoints()
            total_size = 0
            stages = set()
            data_hashes = set()

            for checkpoint in checkpoints:
                total_size += checkpoint.get('file_size_bytes', 0)
                stages.add(checkpoint.get('stage', 'unknown'))
                data_hashes.add(checkpoint.get('data_hash', 'unknown'))

            return {
                'checkpoint_dir': str(self.checkpoint_dir),
                'total_checkpoints': len(checkpoints),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'unique_stages': len(stages),
                'unique_datasets': len(data_hashes),
                'stages': list(stages),
                'oldest_checkpoint': min((c.get('timestamp', '') for c in checkpoints), default=None),
                'newest_checkpoint': max((c.get('timestamp', '') for c in checkpoints), default=None),
                'average_size_mb': (total_size / len(checkpoints) / (1024 * 1024)) if checkpoints else 0,
            }

        except Exception as e:
            warnings.warn(f"Failed to get statistics: {e}", UserWarning)
            return {'error': str(e)}

    def cleanup_old_checkpoints(self, days_old: int = 7) -> int:
        """
        Clean up checkpoints older than specified days

        Parameters
        ----------
        days_old : int
            Delete checkpoints older than this many days

        Returns
        -------
        int
            Number of checkpoints deleted
        """
        deleted_count = 0
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)

        try:
            checkpoints = self.list_checkpoints()
            for checkpoint in checkpoints:
                timestamp_str = checkpoint.get('timestamp', '')
                if timestamp_str:
                    try:
                        checkpoint_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if checkpoint_time.timestamp() < cutoff_time:
                            stage = checkpoint.get('stage')
                            data_hash = checkpoint.get('data_hash')
                            if stage and data_hash and self.delete(stage, data_hash):
                                deleted_count += 1
                    except ValueError:
                        # Skip checkpoints with invalid timestamps
                        continue

        except Exception as e:
            warnings.warn(f"Failed to cleanup old checkpoints: {e}", UserWarning)

        return deleted_count