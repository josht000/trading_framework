"""Console reporting helpers for framework entrypoints."""

import pandas as pd
from rich.console import Console
from rich.table import Table


console = Console()


def section(title):
    console.rule(title)


def table(df, title=None, floatfmt=".2f"):
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    rich_table = Table(title=title, show_lines=False)
    for column in frame.columns:
        rich_table.add_column(str(column))
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        rich_table.add_row(*values)
    console.print(rich_table)
    return rich_table
