class AICError(Exception):
    """Base error for the aic CLI. Its message is shown to the user as-is."""


class ConfigError(AICError):
    """Invalid or missing configuration."""


class ProviderError(AICError):
    """A provider call failed (network, auth, bad response...)."""
