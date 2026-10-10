"""Chunk structures and conversions between the boundary conventions used in this project.

Canonical representation: a partition of the 8 sequence items into consecutive chunks,
written as chunk lengths, e.g. (3, 3, 2) or the name "3-3-2".

Conventions (k is 1-based):

* ``iki_index`` (canonical ground truth): k in 1..7 means IKI_k (the transition from
  item k to item k+1) is a between-chunk transition. (3, 3, 2) -> [3, 6].
  Used by rational_chunking and hcrp_lm.
* ``segment_cp``: k in 1..6 means a change point between IKI k and IKI k+1 (the IKI
  profile is split into segments). Used by community_network, change_point_pelt and
  hsmm_chunking. A single slow boundary IKI_k shows up as change points {k-1, k}.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

N_ITEMS = 8
N_IKIS = N_ITEMS - 1

Partition = Tuple[int, ...]


def parse_structure(name: str) -> Partition:
    """'3-3-2' -> (3, 3, 2); '8' -> (8,)."""
    lengths = tuple(int(x) for x in str(name).strip().split("-") if x)
    validate_partition(lengths)
    return lengths


def structure_name(lengths: Sequence[int]) -> str:
    return "-".join(str(int(x)) for x in lengths)


def validate_partition(lengths: Sequence[int]) -> None:
    if not lengths or any(int(x) < 1 for x in lengths) or sum(lengths) != N_ITEMS:
        raise ValueError(f"Invalid partition {lengths!r}: lengths must be >=1 and sum to {N_ITEMS}.")


def lengths_to_boundaries(lengths: Sequence[int]) -> List[int]:
    """Chunk lengths -> boundary IKI indices (iki_index convention, 1..7)."""
    validate_partition(lengths)
    out: List[int] = []
    pos = 0
    for length in lengths[:-1]:
        pos += int(length)
        out.append(pos)
    return out


def boundaries_to_lengths(boundaries: Iterable[int]) -> Partition:
    """Boundary IKI indices (1..7) -> chunk lengths."""
    bs = sorted({int(b) for b in boundaries if 1 <= int(b) <= N_IKIS})
    edges = [0] + bs + [N_ITEMS]
    return tuple(edges[i + 1] - edges[i] for i in range(len(edges) - 1))


def boundaries_to_item_labels(boundaries: Iterable[int]) -> List[int]:
    """Chunk label for each of the 8 items."""
    bs = {int(b) for b in boundaries}
    labels: List[int] = []
    current = 0
    for item in range(1, N_ITEMS + 1):
        labels.append(current)
        if item in bs:
            current += 1
    return labels


def boundaries_to_segment_cps(boundaries: Iterable[int]) -> List[int]:
    """Translate boundary IKIs into the segment change points an ideal segmenter would report.

    The boundary IKI_k forms its own segment, so the change points are k-1 and k
    (restricted to the valid range 1..6).
    """
    cps = set()
    for k in boundaries:
        k = int(k)
        for c in (k - 1, k):
            if 1 <= c <= N_IKIS - 1:
                cps.add(c)
    return sorted(cps)


def segment_cps_to_iki_labels(cps: Iterable[int]) -> List[int]:
    """Segment label for each of the 7 IKIs given change points (1..6)."""
    cp = {int(c) for c in cps}
    labels: List[int] = []
    current = 0
    for i in range(1, N_IKIS + 1):
        labels.append(current)
        if i in cp:
            current += 1
    return labels


def boundary_iki_mask(boundaries: Iterable[int]) -> List[bool]:
    bs = {int(b) for b in boundaries}
    return [k in bs for k in range(1, N_IKIS + 1)]


def chunk_of_iki(lengths: Sequence[int]) -> List[int]:
    """For each IKI (1..7): index of the chunk it belongs to.

    Within-chunk IKIs belong to their chunk; a boundary IKI is assigned to the chunk it
    initiates (the transition *into* the next chunk).
    """
    item_labels = boundaries_to_item_labels(lengths_to_boundaries(lengths))
    return [item_labels[k] for k in range(1, N_ITEMS)]  # label of item k+1


def position_in_chunk(lengths: Sequence[int]) -> List[float]:
    """Relative position (0..1) of each within-chunk IKI inside its chunk; boundary IKIs -> 0."""
    out: List[float] = []
    bmask = boundary_iki_mask(lengths_to_boundaries(lengths))
    item_labels = boundaries_to_item_labels(lengths_to_boundaries(lengths))
    for k in range(1, N_ITEMS):
        if bmask[k - 1]:
            out.append(0.0)
            continue
        chunk = item_labels[k - 1]
        members = [i for i, lab in enumerate(item_labels, start=1) if lab == chunk]
        n_within = len(members) - 1
        idx = members.index(k)  # IKI k leaves item k
        out.append((idx + 1) / n_within if n_within > 0 else 0.0)
    return out


STRUCTURE_SETS = {
    # Pool used in the static-structure family (includes the no-chunk partition "8").
    "static_pool": ["8", "4-4", "2-2-2-2", "3-3-2", "2-3-3", "3-5", "5-3", "2-6", "2-2-4"],
    # Age-typical chunk sizes (see infos.md): young 4-5 items, old 2-3 items.
    "young": ["4-4", "5-3", "3-5"],
    "old": ["2-2-2-2", "3-3-2", "2-3-3", "2-2-4"],
    "realistic": ["4-4", "3-5", "5-3", "3-3-2", "2-3-3", "2-2-2-2", "2-6", "6-2"],
}
