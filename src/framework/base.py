"""Core interfaces for switchable study pipelines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass
class Setup:
    """A named, rankable strategy setup."""

    label: str
    select: Callable[[pd.DataFrame], pd.DataFrame]
    rank_col: str
    model: Any = None


@dataclass
class RunContext:
    """Inputs and run-local data shared by pipeline stages."""

    cfg: dict
    paths: dict
    args: Any
    flags: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)


class Strategy(ABC):
    """Adapter interface implemented by each study strategy."""

    name: str = ""
    needs: tuple = ()
    exposure_window: dict = {}

    @abstractmethod
    def build_trades(self, ctx: RunContext) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def build_table(self, ctx: RunContext, rebuild: bool) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def setups(self, ctx: RunContext, table: pd.DataFrame) -> dict[str, Setup]:
        raise NotImplementedError

    @abstractmethod
    def rank_experiments(self, ctx: RunContext, table: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def charts(self, ctx: RunContext, table: pd.DataFrame, setups: dict[str, Setup]):
        raise NotImplementedError
