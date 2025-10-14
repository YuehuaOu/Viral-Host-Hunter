"""
- load_taxonomy(excel_path) -> maps
    Load from Excel and return a dictionary `maps`, which contains keys: 'species', 'genes', 'family'.
    Each value is a dict: normalized_name -> set(lineage_strings)
- find_lineage(maps, name, rank=None) -> set
    Search for the lineage of `name` in `maps` (may return multiple, return empty set if not found).

Note: This module does not print or interact with the user, suitable for import as a dependency.
"""

from collections import defaultdict
import pandas as pd

# RANK order (used for concatenating lineage) and corresponding prefix
RANKS = [
    ("phylum", "p"),
    ("class", "c"),
    ("order", "o"),
    ("family", "f"),
    ("genes", "g"),  # column name is 'genes', semantically it is genus
    ("species", "s"),
]


def _normalize_name(name: str) -> str:
    """Internal use: normalize input/table names for consistent matching."""
    if pd.isna(name):
        return ""
    s = str(name).strip()
    # treat underscores as spaces and merge multiple whitespaces
    s = s.replace('_', ' ')
    s = ' '.join(s.split())
    return s.lower()


def load_taxonomy(excel_path: str, sheet_name=0) -> dict:
    """Load from Excel and build maps.

    Returns maps: {'species': {...}, 'genes': {...}, 'family': {...}}
    Each sub-dictionary's value is a set(lineage_string)
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=str)

    maps = {
        'species': defaultdict(set),
        'genes': defaultdict(set),
        'family': defaultdict(set),
    }

    for _, row in df.iterrows():
        # concatenate full lineage (skip missing ranks)
        parts = []
        for col, prefix in RANKS:
            val = row.get(col, '')
            if pd.isna(val) or str(val).strip() == '':
                continue
            parts.append(f"{prefix}__{str(val).strip()}")
        full_lineage = '; '.join(parts)

        # keys
        sp_key = _normalize_name(row.get('species', ''))
        g_key = _normalize_name(row.get('genes', ''))
        f_key = _normalize_name(row.get('family', ''))

        if sp_key:
            maps['species'][sp_key].add(full_lineage)

        if g_key:
            # truncate genus lineage up to g__
            g_parts = []
            for col, prefix in RANKS:
                if col == 'species':
                    break
                val = row.get(col, '')
                if pd.isna(val) or str(val).strip() == '':
                    continue
                g_parts.append(f"{prefix}__{str(val).strip()}")
                if col == 'genes':
                    break
            maps['genes'][g_key].add('; '.join(g_parts))

        if f_key:
            # truncate family lineage up to f__
            f_parts = []
            for col, prefix in RANKS:
                val = row.get(col, '')
                if pd.isna(val) or str(val).strip() == '':
                    continue
                f_parts.append(f"{prefix}__{str(val).strip()}")
                if col == 'family':
                    break
            maps['family'][f_key].add('; '.join(f_parts))

    # convert defaultdict to regular dict for easier serialization/inspection
    maps = {k: dict(v) for k, v in maps.items()}
    return maps


def find_lineage(maps: dict, name: str, rank: str = None) -> set:
    """Find the lineage of `name` in maps.

    Args:
      - maps: dictionary returned by load_taxonomy
      - name: the name to query (can be species/genus/family)
      - rank: optional 'species'/'genes'/'family' to restrict the level; if omitted, will search in priority order: species->genes->family

    Returns: set(lineage_strings) or empty set
    """
    if not name:
        return set()
    key = _normalize_name(name)

    if rank:
        rank = rank.lower()
        if rank not in ('species', 'genes', 'family'):
            raise ValueError("rank must be one of 'species', 'genes', or 'family'")
        # may not exist in maps (return empty set)
        return set(maps.get(rank, {}).get(key, set()))

    # if rank not specified, try by priority
    for r in ('species', 'genes', 'family'):
        vals = maps.get(r, {}).get(key)
        if vals:
            return set(vals)
    return set()
