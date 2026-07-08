"""SEA-RAFT optical-flow point tracker (training-free).

A drop-in replacement for the learned Mamba-3 propagator in the v31 pipeline:
a frozen pretrained SEA-RAFT flow model supplies dense correspondence, which
we chain frame-to-frame to propagate query points. Downstream (DA3 depth
unprojection + TAPVid-3D metrics) is unchanged.
"""

from .model import FlowModel
from .flow_tracker import track_clip, track_clip_with_flow

__all__ = ["FlowModel", "track_clip", "track_clip_with_flow"]
