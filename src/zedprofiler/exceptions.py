"""Custom exception types for ZedProfiler."""


class ZedProfilerError(Exception):
    """Base exception for package-level failures."""


class ContractError(ZedProfilerError):
    """Raised when an input violates a documented data contract."""
