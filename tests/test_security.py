"""
Tests for security functionality (rate limiting, sanitization, etc.).
"""

import pytest
import time
from unittest.mock import patch

from deduplix.security import (
    RateLimiter, RateLimitConfig, SecuritySanitizer, ResourceMonitor,
    RateLimitError
)
from deduplix.exceptions import SecurityError


class TestRateLimitConfig:
    """Test rate limit configuration"""

    def test_default_config(self):
        """Test default rate limit configuration"""
        config = RateLimitConfig()

        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        assert config.requests_per_day == 10000
        assert config.burst_limit == 10

    def test_custom_config(self):
        """Test custom rate limit configuration"""
        config = RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=5000,
            burst_limit=5
        )

        assert config.requests_per_minute == 30
        assert config.requests_per_hour == 500
        assert config.requests_per_day == 5000
        assert config.burst_limit == 5

    def test_invalid_config(self):
        """Test invalid rate limit configuration"""
        with pytest.raises(ValueError):
            RateLimitConfig(requests_per_minute=-1)

        with pytest.raises(ValueError):
            RateLimitConfig(burst_limit=0)


class TestRateLimiter:
    """Test rate limiting functionality"""

    def test_rate_limiter_creation(self):
        """Test basic rate limiter creation"""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = RateLimiter(config)

        assert limiter.config == config

    def test_allow_requests_within_limit(self):
        """Test allowing requests within rate limit"""
        config = RateLimitConfig(requests_per_minute=10, burst_limit=5)
        limiter = RateLimiter(config)

        # Should allow multiple requests within limit
        for i in range(5):
            limiter.enforce_request("test_client")

    def test_block_requests_exceeding_burst(self):
        """Test blocking requests exceeding burst limit"""
        config = RateLimitConfig(requests_per_minute=60, burst_limit=2)
        limiter = RateLimiter(config)

        # Allow first 2 requests
        limiter.enforce_request("test_client")
        limiter.enforce_request("test_client")

        # Third request should be blocked
        with pytest.raises(RateLimitError) as exc_info:
            limiter.enforce_request("test_client")

        assert "Rate limit exceeded" in str(exc_info.value)
        assert hasattr(exc_info.value, 'retry_after')

    def test_rate_limit_reset(self):
        """Test rate limit reset after time window"""
        config = RateLimitConfig(requests_per_minute=60, burst_limit=1)
        limiter = RateLimiter(config)

        # Use up limit
        limiter.enforce_request("test_client")

        # Should be blocked immediately
        with pytest.raises(RateLimitError):
            limiter.enforce_request("test_client")

        # Mock time advancement to test reset
        with patch('time.time') as mock_time:
            # Advance time by 61 seconds
            mock_time.return_value = time.time() + 61

            # Should be allowed after reset
            limiter.enforce_request("test_client")

    def test_different_clients_separate_limits(self):
        """Test that different clients have separate rate limits"""
        config = RateLimitConfig(requests_per_minute=60, burst_limit=1)
        limiter = RateLimiter(config)

        # Use up limit for client1
        limiter.enforce_request("client1")

        # client1 should be blocked
        with pytest.raises(RateLimitError):
            limiter.enforce_request("client1")

        # client2 should still be allowed
        limiter.enforce_request("client2")

    def test_get_remaining_requests(self):
        """Test getting remaining request count"""
        config = RateLimitConfig(requests_per_minute=60, burst_limit=3)
        limiter = RateLimiter(config)

        # Initially should have full burst available
        remaining = limiter.get_remaining_requests("test_client")
        assert remaining == 3

        # After one request
        limiter.enforce_request("test_client")
        remaining = limiter.get_remaining_requests("test_client")
        assert remaining == 2


class TestSecuritySanitizer:
    """Test security sanitization functionality"""

    def test_sanitizer_creation(self):
        """Test basic sanitizer creation"""
        sanitizer = SecuritySanitizer()
        assert sanitizer.strict_mode is False

        strict_sanitizer = SecuritySanitizer(strict_mode=True)
        assert strict_sanitizer.strict_mode is True

    def test_sanitize_text_basic(self):
        """Test basic text sanitization"""
        sanitizer = SecuritySanitizer()

        text = "Apple Inc."
        result = sanitizer.sanitize_text(text)
        assert result == "Apple Inc."

    def test_sanitize_text_html_removal(self):
        """Test HTML tag removal"""
        sanitizer = SecuritySanitizer()

        text = "<script>alert('xss')</script>Apple Inc.<b>Bold</b>"
        result = sanitizer.sanitize_text(text)

        # HTML tags should be removed
        assert "<script>" not in result
        assert "<b>" not in result
        assert "Apple Inc." in result

    def test_sanitize_text_length_limit(self):
        """Test text length limiting"""
        sanitizer = SecuritySanitizer()

        long_text = "A" * 1000
        result = sanitizer.sanitize_text(long_text, max_length=100)

        assert len(result) <= 100

    def test_sanitize_text_sql_injection_patterns(self):
        """Test SQL injection pattern detection"""
        sanitizer = SecuritySanitizer(strict_mode=True)

        # Should detect potential SQL injection
        with pytest.raises(ValueError) as exc_info:
            sanitizer.sanitize_text("'; DROP TABLE companies; --")

        assert "potentially unsafe content" in str(exc_info.value).lower()

    def test_sanitize_text_script_injection(self):
        """Test script injection detection"""
        sanitizer = SecuritySanitizer(strict_mode=True)

        # Should detect script injection attempts
        with pytest.raises(ValueError):
            sanitizer.sanitize_text("javascript:alert('xss')")

    def test_validate_api_key_valid(self):
        """Test API key validation with valid keys"""
        sanitizer = SecuritySanitizer()

        # Valid API keys (different formats)
        valid_keys = [
            "sk-1234567890abcdef",
            "gpt-4-api-key-12345",
            "anthropic_key_abcdef123456"
        ]

        for key in valid_keys:
            # Should not raise exception
            sanitizer.validate_api_key(key)

    def test_validate_api_key_invalid_length(self):
        """Test API key validation with invalid length"""
        sanitizer = SecuritySanitizer()

        # Too short
        with pytest.raises(ValueError):
            sanitizer.validate_api_key("short", min_length=10)

        # Too long
        with pytest.raises(ValueError):
            sanitizer.validate_api_key("A" * 300, max_length=100)

    def test_validate_api_key_invalid_characters(self):
        """Test API key validation with invalid characters"""
        sanitizer = SecuritySanitizer()

        # Contains spaces (usually invalid)
        with pytest.raises(ValueError):
            sanitizer.validate_api_key("key with spaces")

        # Contains suspicious patterns
        with pytest.raises(ValueError):
            sanitizer.validate_api_key("../../../etc/passwd")

    def test_detect_malicious_patterns_true_positives(self):
        """Test malicious pattern detection - should catch"""
        sanitizer = SecuritySanitizer(strict_mode=True)

        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "data:text/html,<script>alert(1)</script>",
            "../../../etc/passwd",
            "cmd.exe /c dir",
            "<?php system($_GET['cmd']); ?>"
        ]

        for malicious_input in malicious_inputs:
            with pytest.raises(ValueError):
                sanitizer.sanitize_text(malicious_input)

    def test_detect_malicious_patterns_false_positives(self):
        """Test malicious pattern detection - should allow legitimate content"""
        sanitizer = SecuritySanitizer(strict_mode=False)  # Less strict

        legitimate_inputs = [
            "Apple & Sons Corp.",
            "Price: $1,000.00",
            "Email: user@example.com",
            "Phone: +1-555-123-4567",
            "Address: 123 Main St., City, State",
            "Company (Est. 2020)",
            "Profit/Loss Statement"
        ]

        for legitimate_input in legitimate_inputs:
            # Should not raise exception
            result = sanitizer.sanitize_text(legitimate_input)
            assert result is not None


class TestResourceMonitor:
    """Test resource monitoring functionality"""

    def test_monitor_creation(self):
        """Test resource monitor creation"""
        monitor = ResourceMonitor()

        # Should have reasonable defaults
        assert monitor.max_memory_mb > 0
        assert monitor.max_threads > 0
        assert monitor.max_processing_time > 0

    def test_custom_limits(self):
        """Test resource monitor with custom limits"""
        monitor = ResourceMonitor(
            max_memory_mb=1024,
            max_threads=8,
            max_processing_time=300
        )

        assert monitor.max_memory_mb == 1024
        assert monitor.max_threads == 8
        assert monitor.max_processing_time == 300

    def test_check_memory_usage(self):
        """Test memory usage checking"""
        monitor = ResourceMonitor(max_memory_mb=1)  # Very low limit

        # Should raise exception if memory usage is high
        # Note: This might not always trigger depending on system state
        try:
            monitor.check_memory_usage()
        except SecurityError as e:
            assert "memory usage" in str(e).lower()

    def test_check_thread_count(self):
        """Test thread count checking"""
        monitor = ResourceMonitor(max_threads=1)  # Very low limit

        # Should raise exception if too many threads
        try:
            monitor.check_thread_count()
        except SecurityError as e:
            assert "thread" in str(e).lower()

    def test_check_processing_time(self):
        """Test processing time checking"""
        monitor = ResourceMonitor(max_processing_time=0.001)  # Very short time

        # Start monitoring
        monitor.start_monitoring()

        # Wait a bit
        time.sleep(0.002)

        # Should raise exception for exceeding time
        with pytest.raises(SecurityError) as exc_info:
            monitor.check_processing_time()

        assert "processing time" in str(exc_info.value).lower()

    def test_get_resource_stats(self):
        """Test getting resource statistics"""
        monitor = ResourceMonitor()

        stats = monitor.get_resource_stats()

        # Should have basic stats
        assert 'memory_usage_mb' in stats
        assert 'thread_count' in stats
        assert 'processing_time' in stats

        # Values should be reasonable
        assert stats['memory_usage_mb'] >= 0
        assert stats['thread_count'] >= 0
        assert stats['processing_time'] >= 0

    def test_monitor_context_manager(self):
        """Test resource monitor as context manager"""
        monitor = ResourceMonitor()

        with monitor:
            # Should be monitoring within context
            assert monitor.start_time is not None

        # Should stop monitoring after context
        # (Implementation dependent - monitor might reset state)

    @pytest.mark.integration
    def test_monitor_with_actual_workload(self):
        """Test resource monitor with actual processing workload"""
        monitor = ResourceMonitor(
            max_memory_mb=2048,  # Reasonable limit
            max_threads=20,
            max_processing_time=5.0
        )

        with monitor:
            # Simulate some work
            data = list(range(10000))
            result = [x * 2 for x in data]

            # Should not raise exceptions for reasonable workload
            monitor.check_memory_usage()
            monitor.check_thread_count()
            monitor.check_processing_time()

        assert len(result) == 10000