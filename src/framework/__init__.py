"""Switchable study framework public API."""

from .base import RunContext, Setup, Strategy
from .registry import get_strategy, register

__all__ = ["RunContext", "Setup", "Strategy", "get_strategy", "register"]
