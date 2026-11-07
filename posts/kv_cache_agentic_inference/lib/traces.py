"""Load AgentX (SemiAnalysis cc-traces-weka) Claude Code traces into one row per model request."""

import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "agentx"
DEFAULT = DATA / "cc-traces-weka-062126-256k" / "traces.jsonl"
BLOCK = 64  # tokens per hash block


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def load(path: Path = DEFAULT) -> pd.DataFrame:
    """One row per request. A stream is the main agent or one sub-agent; each keeps its own KV.

    prefix_hit_tokens: tokens the previous request of the same stream already computed (its
    input hash prefix), i.e. the best case for a cache that never evicts.
    """
    rows = []
    for line in path.open():
        session = json.loads(line)
        streams = [("main", [r for r in session["requests"] if r["type"] != "subagent"])]
        streams += [(g["agent_id"], g["requests"]) for g in session["requests"] if g["type"] == "subagent"]
        for stream, reqs in streams:
            prev = []
            for turn, r in enumerate(reqs):
                rows.append({
                    "session": session["id"],
                    "stream": stream,
                    "turn": turn,
                    "t": r["t"],
                    "isl": r["in"],
                    "osl": r["out"],
                    "api_time": r["api_time"],
                    "think_time": r.get("think_time") or 0.0,
                    "prefix_hit_tokens": _common_prefix(prev, r["hash_ids"]) * BLOCK,
                })
                prev = r["hash_ids"]
    return pd.DataFrame(rows)


def load_cached(path: Path = DEFAULT) -> pd.DataFrame:
    parquet = path.parent.parent / "requests_256k.parquet"
    if not parquet.exists():
        load(path).to_parquet(parquet)
    return pd.read_parquet(parquet)


def subset(df: pd.DataFrame, max_context: int = 131_072) -> pd.DataFrame:
    """Sessions whose every request (input + output) fits max_context, so both models see identical traffic."""
    peak = (df.isl + df.osl).groupby(df.session).max()
    return df[df.session.isin(peak[peak <= max_context].index)]


def parked_profile(df: pd.DataFrame, idle_cap: float) -> dict:
    """Little's-law view of the main-agent streams with every idle gap clipped at idle_cap seconds.

    A request's context stays resident from its start until the next request of the stream,
    so it is weighted by its own service time plus the gap that follows it.
    """
    m = df[df.stream == "main"].sort_values(["session", "turn"])
    gap = m.groupby("session").think_time.shift(-1).fillna(0).clip(upper=idle_cap)
    held = gap + m.api_time
    return {
        "parked_fraction": gap.sum() / held.sum(),
        "context_tokens": ((m.isl + m.osl) * held).sum() / held.sum(),
        "session_seconds": held.groupby(m.session).sum().mean(),
    }
