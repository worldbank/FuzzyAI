"""
Database-based checkpointing implementation.

Stores checkpoints in a SQLite database with data stored as compressed Parquet bytes.
"""

import sqlite3
import json
import time
import gzip
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import warnings
import io

from .base import BaseCheckpointer
from ..exceptions import CheckpointError


class DatabaseCheckpointer(BaseCheckpointer):
    """
    Database-based checkpointing using SQLite.

    Stores checkpoint data as compressed Parquet bytes in a SQLite database
    with rich metadata and querying capabilities.
    """

    def __init__(self, db_path: str = "deduplix_checkpoints.db", compress: bool = True, **kwargs):
        """
        Initialize database checkpointer

        Parameters
        ----------
        db_path : str
            Path to SQLite database file
        compress : bool
            Whether to compress checkpoint data
        **kwargs
            Additional configuration parameters
        """
        super().__init__(kwargs)
        self.db_path = db_path
        self.compress = compress
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stage TEXT NOT NULL,
                        data_hash TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        rows INTEGER,
                        columns TEXT,  -- JSON array of column names
                        data_bytes BLOB,  -- Compressed Parquet data
                        metadata TEXT,  -- JSON metadata
                        file_size_bytes INTEGER,
                        compressed BOOLEAN DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(stage, data_hash)
                    )
                """)

                # Create indexes for better performance
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stage_hash
                    ON checkpoints(stage, data_hash)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON checkpoints(timestamp)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_data_hash
                    ON checkpoints(data_hash)
                """)

        except Exception as e:
            raise CheckpointError(
                f"Failed to initialize database: {e}",
                checkpoint_path=self.db_path
            ) from e

    def save(
        self,
        data: pd.DataFrame,
        stage: str,
        data_hash: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save checkpoint data to database

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
            self.validate_stage_name(stage)
            self.validate_data_hash(data_hash)

            # Convert DataFrame to Parquet bytes
            parquet_buffer = io.BytesIO()
            data.to_parquet(parquet_buffer, index=False)
            parquet_bytes = parquet_buffer.getvalue()

            # Compress if enabled
            if self.compress:
                parquet_bytes = gzip.compress(parquet_bytes)

            # Prepare metadata
            checkpoint_metadata = {
                'stage': stage,
                'data_hash': data_hash,
                'timestamp': datetime.now().isoformat(),
                'rows': len(data),
                'columns': list(data.columns),
                'compressed': self.compress,
                'custom_metadata': metadata or {}
            }

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO checkpoints
                    (stage, data_hash, timestamp, rows, columns, data_bytes, metadata,
                     file_size_bytes, compressed, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    stage,
                    data_hash,
                    datetime.now().isoformat(),
                    len(data),
                    json.dumps(list(data.columns)),
                    parquet_bytes,
                    json.dumps(checkpoint_metadata),
                    len(parquet_bytes),
                    self.compress,
                    datetime.now().isoformat()
                ))

        except Exception as e:
            raise CheckpointError(
                f"Failed to save checkpoint for stage '{stage}': {e}",
                checkpoint_stage=stage,
                checkpoint_path=self.db_path,
                context={'data_hash': data_hash, 'data_shape': data.shape}
            ) from e

    def load(self, stage: str, data_hash: str) -> Optional[pd.DataFrame]:
        """
        Load checkpoint data from database

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
            self.validate_stage_name(stage)
            self.validate_data_hash(data_hash)

            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("""
                    SELECT data_bytes, compressed
                    FROM checkpoints
                    WHERE stage = ? AND data_hash = ?
                """, (stage, data_hash)).fetchone()

                if result:
                    data_bytes, compressed = result

                    # Decompress if needed
                    if compressed:
                        data_bytes = gzip.decompress(data_bytes)

                    # Convert back to DataFrame
                    parquet_buffer = io.BytesIO(data_bytes)
                    return pd.read_parquet(parquet_buffer)

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
            self.validate_stage_name(stage)
            self.validate_data_hash(data_hash)

            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("""
                    SELECT COUNT(*) FROM checkpoints
                    WHERE stage = ? AND data_hash = ?
                """, (stage, data_hash)).fetchone()

                return result[0] > 0 if result else False

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
            with sqlite3.connect(self.db_path) as conn:
                # Set row factory to return dictionaries
                conn.row_factory = sqlite3.Row

                if data_hash:
                    results = conn.execute("""
                        SELECT stage, data_hash, timestamp, rows, columns,
                               file_size_bytes, compressed, created_at, metadata
                        FROM checkpoints
                        WHERE data_hash = ?
                        ORDER BY timestamp DESC
                    """, (data_hash,)).fetchall()
                else:
                    results = conn.execute("""
                        SELECT stage, data_hash, timestamp, rows, columns,
                               file_size_bytes, compressed, created_at, metadata
                        FROM checkpoints
                        ORDER BY timestamp DESC
                    """).fetchall()

                for row in results:
                    checkpoint_info = dict(row)
                    # Parse JSON fields
                    if checkpoint_info['columns']:
                        checkpoint_info['columns'] = json.loads(checkpoint_info['columns'])
                    if checkpoint_info['metadata']:
                        checkpoint_info['metadata'] = json.loads(checkpoint_info['metadata'])

                    checkpoints.append(checkpoint_info)

        except Exception as e:
            warnings.warn(f"Failed to list checkpoints: {e}", UserWarning)

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
            self.validate_stage_name(stage)
            self.validate_data_hash(data_hash)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM checkpoints
                    WHERE stage = ? AND data_hash = ?
                """, (stage, data_hash))

                return cursor.rowcount > 0

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
        try:
            with sqlite3.connect(self.db_path) as conn:
                if data_hash:
                    cursor = conn.execute("""
                        DELETE FROM checkpoints
                        WHERE data_hash = ?
                    """, (data_hash,))
                else:
                    cursor = conn.execute("DELETE FROM checkpoints")

                return cursor.rowcount

        except Exception as e:
            warnings.warn(f"Failed to clear checkpoints: {e}", UserWarning)
            return 0

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
            self.validate_stage_name(stage)
            self.validate_data_hash(data_hash)

            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("""
                    SELECT metadata FROM checkpoints
                    WHERE stage = ? AND data_hash = ?
                """, (stage, data_hash)).fetchone()

                if result and result[0]:
                    return json.loads(result[0])

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
            with sqlite3.connect(self.db_path) as conn:
                # Basic counts
                stats = conn.execute("""
                    SELECT
                        COUNT(*) as total_checkpoints,
                        SUM(file_size_bytes) as total_size_bytes,
                        COUNT(DISTINCT stage) as unique_stages,
                        COUNT(DISTINCT data_hash) as unique_datasets,
                        AVG(file_size_bytes) as avg_size_bytes,
                        MIN(timestamp) as oldest_checkpoint,
                        MAX(timestamp) as newest_checkpoint,
                        SUM(CASE WHEN compressed THEN 1 ELSE 0 END) as compressed_count
                    FROM checkpoints
                """).fetchone()

                if stats:
                    total_checkpoints, total_size, unique_stages, unique_datasets, \
                    avg_size, oldest, newest, compressed_count = stats

                    # Get stage breakdown
                    stage_counts = conn.execute("""
                        SELECT stage, COUNT(*) as count, SUM(file_size_bytes) as size_bytes
                        FROM checkpoints
                        GROUP BY stage
                        ORDER BY count DESC
                    """).fetchall()

                    return {
                        'db_path': self.db_path,
                        'total_checkpoints': total_checkpoints or 0,
                        'total_size_bytes': total_size or 0,
                        'total_size_mb': (total_size or 0) / (1024 * 1024),
                        'unique_stages': unique_stages or 0,
                        'unique_datasets': unique_datasets or 0,
                        'average_size_mb': (avg_size or 0) / (1024 * 1024),
                        'oldest_checkpoint': oldest,
                        'newest_checkpoint': newest,
                        'compressed_checkpoints': compressed_count or 0,
                        'compression_enabled': self.compress,
                        'stages': [{'stage': stage, 'count': count, 'size_mb': size_bytes / (1024 * 1024)}
                                 for stage, count, size_bytes in stage_counts],
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
        try:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM checkpoints
                    WHERE timestamp < ?
                """, (cutoff_iso,))

                return cursor.rowcount

        except Exception as e:
            warnings.warn(f"Failed to cleanup old checkpoints: {e}", UserWarning)
            return 0

    def optimize_database(self) -> None:
        """Optimize the database by running VACUUM and ANALYZE"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
        except Exception as e:
            warnings.warn(f"Failed to optimize database: {e}", UserWarning)

    def export_checkpoints(self, export_path: str, data_hash: Optional[str] = None) -> int:
        """
        Export checkpoints to a directory as individual files

        Parameters
        ----------
        export_path : str
            Directory to export checkpoints to
        data_hash : Optional[str]
            Export only checkpoints for specific dataset hash

        Returns
        -------
        int
            Number of checkpoints exported
        """
        from pathlib import Path

        try:
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)

            checkpoints = self.list_checkpoints(data_hash)
            exported_count = 0

            for checkpoint in checkpoints:
                stage = checkpoint['stage']
                hash_val = checkpoint['data_hash']

                # Load and save as Parquet file
                df = self.load(stage, hash_val)
                if df is not None:
                    filename = f"{stage}_{hash_val}.parquet"
                    df.to_parquet(export_dir / filename, index=False)

                    # Also export metadata
                    meta_filename = f"{stage}_{hash_val}.meta.json"
                    with open(export_dir / meta_filename, 'w') as f:
                        json.dump(checkpoint['metadata'], f, indent=2)

                    exported_count += 1

            return exported_count

        except Exception as e:
            warnings.warn(f"Failed to export checkpoints: {e}", UserWarning)
            return 0