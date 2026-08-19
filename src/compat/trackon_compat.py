"""Compatibility patches that let the vendored Track-On run on transformers v5.

Track-On pins transformers==4.56.1, where `DINOv3ViTModel` exposed its
transformer blocks directly as `.layer`. From v5 the encoder is nested
(`self.model = DINOv3ViTEncoder(config)`), so the blocks live at `.model.layer`
and Track-On's adapter raises:

    AttributeError: 'DINOv3ViTModel' object has no attribute 'layer'

rfdetr requires transformers>=5.1, so pinning back is not an option. Rather than
editing external/track_on -- which is gitignored, is its own git repo, and would
lose the change on any re-clone -- restore the old attribute on the transformers
class itself. Track-On's code then works unmodified.

Call `apply()` before constructing the Track-On Predictor.
"""

_APPLIED = False


def _dinov3_layer(self):
    """Blocks of a DINOv3 encoder, under either transformers layout."""
    modules = self._modules
    if 'layer' in modules:          # transformers <= 4.56
        return modules['layer']
    return modules['model'].layer   # transformers >= 5


def apply():
    """Idempotently add `DINOv3ViTModel.layer` when transformers v5 removed it.

    Returns True if the patch is in place (or was already unnecessary), False if
    transformers' DINOv3 module could not be located at all.
    """
    global _APPLIED
    if _APPLIED:
        return True

    try:
        from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
    except ImportError:
        return False

    # Only patch when the class itself does not already define `layer`. On
    # transformers <= 4.56 `layer` is an instance submodule rather than a class
    # attribute, so the property is still the right thing to install -- it falls
    # back to the instance's own ModuleList.
    if not isinstance(getattr(DINOv3ViTModel, 'layer', None), property):
        DINOv3ViTModel.layer = property(_dinov3_layer)

    _APPLIED = True
    return True
