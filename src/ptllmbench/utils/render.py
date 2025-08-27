from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

def render_header(console: Console, torch_version: str, transformers_version: str, backend: str, device: str) -> None:
    console.print(Panel.fit(f'pytorch-llm-bench\n[bold]torch[/]: {torch_version}  [bold]transformers[/]: {transformers_version}\n[bold]backend[/]: {backend}\n[bold]device[/]: {device}', title='env', box=box.ROUNDED))

def render_table(console: Console, rows: List[Dict[str,str]]) -> None:
    t = Table(box=box.SIMPLE_HEAVY)
    for col in ["model","params","dtype","quant","backend","device","batch","test","t/s"]:
        t.add_column(col)
    for r in rows:
        t.add_row(r.get('model',''), r.get('params',''), r.get('dtype',''), r.get('quant',''), r.get('backend',''), r.get('device',''), str(r.get('batch','')), r.get('test',''), r.get('tps',''))
    console.print(t)

def render_footer(console: Console, torch_version: str, transformers_version: str) -> None:
    console.print(f'build: torch {torch_version}, transformers {transformers_version}')
