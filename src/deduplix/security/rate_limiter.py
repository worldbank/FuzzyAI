"""
Rate limiting implementation for API calls and resource usage.

Provides configurable rate limiting to prevent abuse and ensure
stable operation under load.
"""

import time
import threading
from typing import Dict, Optional, Union
from collections import defaultdict, deque
from dataclasses import dataclass
import warnings


class RateLimitError(Exception):
    """Raised when rate limit is exceeded"""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Max requests in burst
    burst_window: float = 10.0  # Burst window in seconds


class RateLimiter:
    """
    Thread-safe rate limiter with multiple time windows and burst protection.

    Supports rate limiting across multiple time windows (minute, hour, day)
    and provides burst protection to prevent sudden spikes.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter

        Parameters
        ----------
        config : Optional[RateLimitConfig]
            Rate limiting configuration. Uses defaults if None.
        """
        self.config = config or RateLimitConfig()
        self._lock = threading.RLock()

        # Request timestamps for different windows
        self._minute_requests = deque()  # Last 60 seconds
        self._hour_requests = deque()    # Last 3600 seconds
        self._day_requests = deque()     # Last 86400 seconds
        self._burst_requests = deque()   # Burst window

        # Statistics
        self._total_requests = 0
        self._total_blocked = 0
        self._last_request_time = None

    def allow_request(self, identifier: str = "default") -> bool:
        """
        Check if request is allowed under current rate limits

        Parameters
        ----------
        identifier : str
            Identifier for the request source (e.g., API key, user ID)

        Returns
        -------
        bool
            True if request is allowed, False if rate limited
        """
        with self._lock:
            now = time.time()

            # Clean old requests from all windows
            self._clean_old_requests(now)

            # Check all rate limits
            if not self._check_minute_limit():
                self._total_blocked += 1
                return False

            if not self._check_hour_limit():
                self._total_blocked += 1
                return False

            if not self._check_day_limit():
                self._total_blocked += 1
                return False

            if not self._check_burst_limit():
                self._total_blocked += 1
                return False

            # All checks passed - record the request
            self._record_request(now)
            return True

    def enforce_request(self, identifier: str = "default") -> None:
        """
        Enforce rate limiting - raises exception if limit exceeded

        Parameters
        ----------
        identifier : str
            Identifier for the request source

        Raises
        ------
        RateLimitError
            If rate limit is exceeded
        """
        if not self.allow_request(identifier):
            retry_after = self.get_retry_after()
            raise RateLimitError(
                f"Rate limit exceeded for '{identifier}'. "
                f"Try again in {retry_after:.1f} seconds.",
                retry_after=retry_after
            )

    def get_retry_after(self) -> float:
        """
        Get recommended retry delay in seconds

        Returns
        -------
        float
            Seconds to wait before next request
        """
        with self._lock:
            now = time.time()
            self._clean_old_requests(now)

            # Check which limit is hit and return appropriate delay
            if len(self._burst_requests) >= self.config.burst_limit:
                # Wait for burst window to clear
                oldest_burst = self._burst_requests[0]
                return max(0, self.config.burst_window - (now - oldest_burst))

            if len(self._minute_requests) >= self.config.requests_per_minute:
                # Wait for oldest request in minute window to expire
                oldest_minute = self._minute_requests[0]
                return max(0, 60 - (now - oldest_minute))

            if len(self._hour_requests) >= self.config.requests_per_hour:
                # Wait for oldest request in hour window to expire
                oldest_hour = self._hour_requests[0]
                return max(0, 3600 - (now - oldest_hour))

            if len(self._day_requests) >= self.config.requests_per_day:
                # Wait for oldest request in day window to expire
                oldest_day = self._day_requests[0]
                return max(0, 86400 - (now - oldest_day))

            return 0.0

    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get rate limiter statistics

        Returns
        -------
        Dict[str, Union[int, float]]
            Statistics including total requests, blocked requests, etc.
        """
        with self._lock:
            now = time.time()
            self._clean_old_requests(now)

            return {
                'total_requests': self._total_requests,
                'total_blocked': self._total_blocked,
                'current_minute_requests': len(self._minute_requests),
                'current_hour_requests': len(self._hour_requests),
                'current_day_requests': len(self._day_requests),
                'current_burst_requests': len(self._burst_requests),
                'block_rate': self._total_blocked / max(1, self._total_requests),
                'last_request_time': self._last_request_time,
                'time_since_last_request': now - self._last_request_time if self._last_request_time else None
            }

    def reset(self) -> None:
        """Reset all rate limiting counters"""
        with self._lock:
            self._minute_requests.clear()
            self._hour_requests.clear()
            self._day_requests.clear()
            self._burst_requests.clear()
            self._total_requests = 0
            self._total_blocked = 0
            self._last_request_time = None

    def _clean_old_requests(self, now: float) -> None:
        """Remove requests outside their respective time windows"""
        # Clean minute window (60 seconds)
        while self._minute_requests and (now - self._minute_requests[0]) > 60:
            self._minute_requests.popleft()

        # Clean hour window (3600 seconds)
        while self._hour_requests and (now - self._hour_requests[0]) > 3600:
            self._hour_requests.popleft()

        # Clean day window (86400 seconds)
        while self._day_requests and (now - self._day_requests[0]) > 86400:
            self._day_requests.popleft()

        # Clean burst window
        while self._burst_requests and (now - self._burst_requests[0]) > self.config.burst_window:
            self._burst_requests.popleft()

    def _check_minute_limit(self) -> bool:
        """Check if within minute rate limit"""
        return len(self._minute_requests) < self.config.requests_per_minute

    def _check_hour_limit(self) -> bool:
        """Check if within hour rate limit"""
        return len(self._hour_requests) < self.config.requests_per_hour

    def _check_day_limit(self) -> bool:
        """Check if within day rate limit"""
        return len(self._day_requests) < self.config.requests_per_day

    def _check_burst_limit(self) -> bool:
        """Check if within burst rate limit"""
        return len(self._burst_requests) < self.config.burst_limit

    def _record_request(self, timestamp: float) -> None:
        """Record a successful request in all windows"""
        self._minute_requests.append(timestamp)
        self._hour_requests.append(timestamp)
        self._day_requests.append(timestamp)
        self._burst_requests.append(timestamp)
        self._total_requests += 1
        self._last_request_time = timestamp


class GlobalRateLimiter:
    """
    Global rate limiter that manages multiple rate limiters by identifier.

    Useful for rate limiting different API keys, users, or services
    with different limits.
    """

    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._configs: Dict[str, RateLimitConfig] = {}
        self._lock = threading.RLock()

    def set_rate_limit(self, identifier: str, config: RateLimitConfig) -> None:
        """
        Set rate limit configuration for a specific identifier

        Parameters
        ----------
        identifier : str
            Identifier to set rate limit for
        config : RateLimitConfig
            Rate limit configuration
        """
        with self._lock:
            self._configs[identifier] = config
            if identifier in self._limiters:
                # Update existing limiter with new config
                self._limiters[identifier] = RateLimiter(config)

    def allow_request(self, identifier: str = "default") -> bool:
        """
        Check if request is allowed for the given identifier

        Parameters
        ----------
        identifier : str
            Identifier for the request source

        Returns
        -------
        bool
            True if request is allowed, False if rate limited
        """
        with self._lock:
            if identifier not in self._limiters:
                config = self._configs.get(identifier, RateLimitConfig())
                self._limiters[identifier] = RateLimiter(config)

            return self._limiters[identifier].allow_request(identifier)

    def enforce_request(self, identifier: str = "default") -> None:
        """
        Enforce rate limiting for the given identifier

        Parameters
        ----------
        identifier : str
            Identifier for the request source

        Raises
        ------
        RateLimitError
            If rate limit is exceeded
        """
        with self._lock:
            if identifier not in self._limiters:
                config = self._configs.get(identifier, RateLimitConfig())
                self._limiters[identifier] = RateLimiter(config)

            self._limiters[identifier].enforce_request(identifier)

    def get_statistics(self, identifier: Optional[str] = None) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Get statistics for rate limiters

        Parameters
        ----------
        identifier : Optional[str]
            Specific identifier to get stats for. If None, returns all.

        Returns
        -------
        Dict[str, Dict[str, Union[int, float]]]
            Statistics by identifier
        """
        with self._lock:
            if identifier and identifier in self._limiters:
                return {identifier: self._limiters[identifier].get_statistics()}

            return {
                id_: limiter.get_statistics()
                for id_, limiter in self._limiters.items()
            }

    def reset(self, identifier: Optional[str] = None) -> None:
        """
        Reset rate limiter counters

        Parameters
        ----------
        identifier : Optional[str]
            Specific identifier to reset. If None, resets all.
        """
        with self._lock:
            if identifier and identifier in self._limiters:
                self._limiters[identifier].reset()
            else:
                for limiter in self._limiters.values():
                    limiter.reset()


# Global instance for convenience
_global_rate_limiter = GlobalRateLimiter()

def get_global_rate_limiter() -> GlobalRateLimiter:
    """Get the global rate limiter instance"""
    return _global_rate_limiter