"""Minimal stand-in for the parts of mmcv that Track-On uses.

Track-On imports exactly one symbol from mmcv (`MultiScaleDeformableAttention`),
but mmcv ships it as a compiled CUDA op. mmcv's last release (v2.2.0, Apr 2024)
predates Blackwell (sm_120) and modern torch, so it cannot be built here. This
package provides a pure-PyTorch replacement instead.

It is placed at the END of sys.path (see src/main.py), so a real mmcv install
takes precedence if one is ever available.
"""

__all__ = ['ops']
