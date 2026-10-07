"""
Find Keras layers in a SavedModel by their variable names.

Keras names the first layer of a kind without a suffix (conv1d, dense) and the
later ones with one (conv1d_1, conv1d_192). Both forms are handled here, and
layers come back in the order they were created.
"""

import re


def layer_names(V, kind):
    """Layer names such as 'conv1d', 'conv1d_1', ... that own a kernel,
    in model order."""
    pat = re.compile(rf"^({re.escape(kind)}(?:_(\d+))?)/kernel:0$")
    found = []
    for k in V:
        m = pat.match(k)
        if m:
            index = int(m.group(2)) if m.group(2) else -1
            found.append((index, m.group(1)))
    return [name for _, name in sorted(found)]


def batchnorm_for(name, kind):
    """The BatchNorm sharing a layer's suffix: conv1d_192 -> batch_normalization_192."""
    return "batch_normalization" + name[len(kind):]