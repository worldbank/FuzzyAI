"""
Resource monitoring and limits for security and stability.

Provides monitoring and limiting of resource usage including:
- Memory usage tracking
- Thread pool limits
- Processing time limits
- Disk space monitoring
"""

import time
import psutil
import threading
import os
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings


@dataclass
class ResourceLimits:
    """Configuration for resource limits"""
    max_memory_mb: int = 1024  # Maximum memory usage in MB
    max_threads: int = 8       # Maximum number of threads
    max_processing_time: int = 3600  # Maximum processing time in seconds
    max_disk_usage_percent: float = 90.0  # Maximum disk usage percentage
    check_interval: float = 5.0  # Resource check interval in seconds


class ResourceMonitor:
    """
    Monitor and enforce resource limits for security and stability.

    Tracks memory usage, thread counts, processing time, and disk usage
    to prevent resource exhaustion attacks and ensure stable operation.
    """

    def __init__(self, limits: Optional[ResourceLimits] = None):
        """
        Initialize resource monitor

        Parameters
        ----------
        limits : Optional[ResourceLimits]
            Resource limits configuration. Uses defaults if None.
        """
        self.limits = limits or ResourceLimits()
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.RLock()
        self._callbacks = []
        self._process = psutil.Process()

        # Statistics
        self._stats = {
            'peak_memory_mb': 0.0,
            'peak_threads': 0,
            'total_violations': 0,
            'memory_violations': 0,
            'thread_violations': 0,
            'time_violations': 0,
            'disk_violations': 0,
            'start_time': time.time(),
        }

    def start_monitoring(self) -> None:
        """Start background resource monitoring"""
        with self._lock:
            if not self._monitoring:
                self._monitoring = True
                self._monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    daemon=True,
                    name="ResourceMonitor"
                )
                self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop background resource monitoring"""
        with self._lock:
            self._monitoring = False
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=1.0)

    def add_violation_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Add callback to be called when resource violations occur

        Parameters
        ----------
        callback : Callable[[str, Dict[str, Any]], None]
            Callback function that takes violation type and details
        """
        with self._lock:
            self._callbacks.append(callback)

    def check_memory_usage(self) -> Dict[str, Any]:
        """
        Check current memory usage

        Returns
        -------
        Dict[str, Any]
            Memory usage information and violation status
        """
        try:
            memory_info = self._process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = self._process.memory_percent()

            # Update peak memory
            if memory_mb > self._stats['peak_memory_mb']:
                self._stats['peak_memory_mb'] = memory_mb

            result = {
                'memory_mb': memory_mb,
                'memory_percent': memory_percent,
                'limit_mb': self.limits.max_memory_mb,
                'violation': memory_mb > self.limits.max_memory_mb,
                'available_mb': self.limits.max_memory_mb - memory_mb
            }

            if result['violation']:
                self._stats['memory_violations'] += 1
                self._stats['total_violations'] += 1
                self._trigger_callbacks('memory_violation', result)

            return result

        except Exception as e:
            warnings.warn(f"Failed to check memory usage: {e}", UserWarning)
            return {'error': str(e)}

    def check_thread_usage(self) -> Dict[str, Any]:
        """
        Check current thread usage

        Returns
        -------
        Dict[str, Any]
            Thread usage information and violation status
        """
        try:
            thread_count = self._process.num_threads()

            # Update peak threads
            if thread_count > self._stats['peak_threads']:
                self._stats['peak_threads'] = thread_count

            result = {
                'thread_count': thread_count,
                'limit': self.limits.max_threads,
                'violation': thread_count > self.limits.max_threads,
                'available': self.limits.max_threads - thread_count
            }

            if result['violation']:
                self._stats['thread_violations'] += 1
                self._stats['total_violations'] += 1
                self._trigger_callbacks('thread_violation', result)

            return result

        except Exception as e:
            warnings.warn(f"Failed to check thread usage: {e}", UserWarning)
            return {'error': str(e)}

    def check_processing_time(self, start_time: float) -> Dict[str, Any]:
        """
        Check processing time against limits

        Parameters
        ----------
        start_time : float
            Processing start time (from time.time())

        Returns
        -------
        Dict[str, Any]
            Processing time information and violation status
        """
        current_time = time.time()
        elapsed_time = current_time - start_time

        result = {
            'elapsed_seconds': elapsed_time,
            'limit_seconds': self.limits.max_processing_time,
            'violation': elapsed_time > self.limits.max_processing_time,
            'remaining_seconds': self.limits.max_processing_time - elapsed_time
        }

        if result['violation']:
            self._stats['time_violations'] += 1
            self._stats['total_violations'] += 1
            self._trigger_callbacks('time_violation', result)

        return result

    def check_disk_usage(self, path: str = '.') -> Dict[str, Any]:
        """
        Check disk usage at specified path

        Parameters
        ----------
        path : str
            Path to check disk usage for

        Returns
        -------
        Dict[str, Any]
            Disk usage information and violation status
        """
        try:
            disk_usage = psutil.disk_usage(path)
            usage_percent = (disk_usage.used / disk_usage.total) * 100

            result = {
                'path': path,
                'used_gb': disk_usage.used / (1024**3),
                'total_gb': disk_usage.total / (1024**3),
                'free_gb': disk_usage.free / (1024**3),
                'usage_percent': usage_percent,
                'limit_percent': self.limits.max_disk_usage_percent,
                'violation': usage_percent > self.limits.max_disk_usage_percent
            }

            if result['violation']:
                self._stats['disk_violations'] += 1
                self._stats['total_violations'] += 1
                self._trigger_callbacks('disk_violation', result)

            return result

        except Exception as e:
            warnings.warn(f"Failed to check disk usage: {e}", UserWarning)
            return {'error': str(e)}

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information

        Returns
        -------
        Dict[str, Any]
            System information including CPU, memory, disk
        """
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'cpu_count': psutil.cpu_count(),
                'memory': dict(psutil.virtual_memory()._asdict()),
                'disk': dict(psutil.disk_usage('.')._asdict()),
                'process_pid': os.getpid(),
                'process_threads': self._process.num_threads(),
                'process_memory_mb': self._process.memory_info().rss / 1024 / 1024,
                'process_cpu_percent': self._process.cpu_percent()
            }
        except Exception as e:
            warnings.warn(f"Failed to get system info: {e}", UserWarning)
            return {'error': str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get resource monitoring statistics

        Returns
        -------
        Dict[str, Any]
            Monitoring statistics and current usage
        """
        current_memory = self.check_memory_usage()
        current_threads = self.check_thread_usage()

        return {
            **self._stats,
            'monitoring_active': self._monitoring,
            'uptime_seconds': time.time() - self._stats['start_time'],
            'current_memory_mb': current_memory.get('memory_mb', 0),
            'current_threads': current_threads.get('thread_count', 0),
            'limits': {
                'max_memory_mb': self.limits.max_memory_mb,
                'max_threads': self.limits.max_threads,
                'max_processing_time': self.limits.max_processing_time,
                'max_disk_usage_percent': self.limits.max_disk_usage_percent
            }
        }

    def create_limited_executor(self, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """
        Create a ThreadPoolExecutor with resource limits

        Parameters
        ----------
        max_workers : Optional[int]
            Maximum number of worker threads. Uses limit if None.

        Returns
        -------
        ThreadPoolExecutor
            Limited thread pool executor
        """
        if max_workers is None:
            max_workers = min(self.limits.max_threads, 8)

        max_workers = min(max_workers, self.limits.max_threads)

        return ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ResourceLimited"
        )

    def enforce_limits(self, check_memory: bool = True, check_threads: bool = True) -> None:
        """
        Enforce resource limits and raise exceptions if violated

        Parameters
        ----------
        check_memory : bool
            Whether to check memory limits
        check_threads : bool
            Whether to check thread limits

        Raises
        ------
        ResourceError
            If resource limits are exceeded
        """
        if check_memory:
            memory_check = self.check_memory_usage()
            if memory_check.get('violation', False):
                raise ResourceError(
                    f"Memory limit exceeded: {memory_check['memory_mb']:.1f}MB "
                    f"(limit: {self.limits.max_memory_mb}MB)"
                )

        if check_threads:
            thread_check = self.check_thread_usage()
            if thread_check.get('violation', False):
                raise ResourceError(
                    f"Thread limit exceeded: {thread_check['thread_count']} "
                    f"(limit: {self.limits.max_threads})"
                )

    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while self._monitoring:
            try:
                # Check all resources
                self.check_memory_usage()
                self.check_thread_usage()

                time.sleep(self.limits.check_interval)

            except Exception as e:
                warnings.warn(f"Error in resource monitoring loop: {e}", UserWarning)
                time.sleep(1.0)

    def _trigger_callbacks(self, violation_type: str, details: Dict[str, Any]) -> None:
        """Trigger violation callbacks"""
        for callback in self._callbacks:
            try:
                callback(violation_type, details)
            except Exception as e:
                warnings.warn(f"Error in violation callback: {e}", UserWarning)


class ResourceError(Exception):
    """Raised when resource limits are exceeded"""
    pass


# Context manager for resource monitoring
class ResourceMonitorContext:
    """Context manager for resource monitoring during operations"""

    def __init__(self, monitor: ResourceMonitor, check_on_enter: bool = True, check_on_exit: bool = True):
        self.monitor = monitor
        self.check_on_enter = check_on_enter
        self.check_on_exit = check_on_exit
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        if self.check_on_enter:
            self.monitor.enforce_limits()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.check_on_exit:
            self.monitor.enforce_limits()

        # Check processing time
        if self.start_time:
            time_check = self.monitor.check_processing_time(self.start_time)
            if time_check.get('violation', False):
                warnings.warn(
                    f"Processing time limit exceeded: {time_check['elapsed_seconds']:.1f}s "
                    f"(limit: {self.monitor.limits.max_processing_time}s)",
                    UserWarning
                )