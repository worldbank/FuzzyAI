"""
Abstract base class for checkpointing implementations.

Defines the interface that all checkpointing backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd
from pathlib import Path


class BaseCheckpointer(ABC):
    """
    Abstract base class for checkpointing implementations.

    Defines the interface for saving and loading intermediate results
    during deduplication operations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize checkpointer

        Parameters
        ----------
        config : Optional[Dict[str, Any]]
            Configuration parameters specific to the implementation
        """
        self.config = config or {}

    @abstractmethod
    def save(self, data: pd.DataFrame, stage: str, data_hash: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save checkpoint data

        Parameters
        ----------
        data : pd.DataFrame
            Data to checkpoint
        stage : str
            Processing stage identifier (e.g., 'matching', 'validation')
        data_hash : str
            Unique hash identifying the dataset
        metadata : Optional[Dict[str, Any]]
            Additional metadata to store with the checkpoint
        """
        pass

    @abstractmethod
    def load(self, stage: str, data_hash: str) -> Optional[pd.DataFrame]:
        """
        Load checkpoint data

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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def list_checkpoints(self, data_hash: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available checkpoints

        Parameters
        ----------
        data_hash : Optional[str]
            Filter by specific dataset hash. If None, list all checkpoints.

        Returns
        -------
        List[Dict[str, Any]]
            List of checkpoint information dictionaries
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def clear(self, data_hash: Optional[str] = None) -> int:
        """
        Clear checkpoints

        Parameters
        ----------
        data_hash : Optional[str]
            Clear only checkpoints for specific dataset hash.
            If None, clear all checkpoints.

        Returns
        -------
        int
            Number of checkpoints deleted
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get checkpointing system statistics

        Returns
        -------
        Dict[str, Any]
            Statistics about checkpoint usage, storage, etc.
        """
        pass

    def validate_stage_name(self, stage: str) -> None:
        """
        Validate stage name format

        Parameters
        ----------
        stage : str
            Stage name to validate

        Raises
        ------
        ValueError
            If stage name is invalid
        """
        if not stage or not isinstance(stage, str):
            raise ValueError("Stage name must be a non-empty string")

        # Allow alphanumeric characters, underscores, and hyphens
        if not all(c.isalnum() or c in '_-' for c in stage):
            raise ValueError(f"Invalid stage name: {stage}. Use only alphanumeric characters, underscores, and hyphens.")

    def validate_data_hash(self, data_hash: str) -> None:
        """
        Validate data hash format

        Parameters
        ----------
        data_hash : str
            Data hash to validate

        Raises
        ------
        ValueError
            If data hash is invalid
        """
        if not data_hash or not isinstance(data_hash, str):
            raise ValueError("Data hash must be a non-empty string")

        # Should be hexadecimal characters
        if not all(c in '0123456789abcdefABCDEF_' for c in data_hash):
            raise ValueError(f"Invalid data hash format: {data_hash}")

    def create_checkpoint_id(self, stage: str, data_hash: str) -> str:
        """
        Create a unique checkpoint identifier

        Parameters
        ----------
        stage : str
            Processing stage identifier
        data_hash : str
            Unique hash identifying the dataset

        Returns
        -------
        str
            Unique checkpoint identifier
        """
        self.validate_stage_name(stage)
        self.validate_data_hash(data_hash)
        return f"{stage}_{data_hash}"