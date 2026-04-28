# Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Helpers for configuring the TensorFlow MUSA graph optimizer."""

MUSA_GRAPH_OPTIMIZER_NAME = "musa_graph_optimizer"


def _get_config_proto_class():
    import tensorflow as tf

    if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
        return tf.compat.v1.ConfigProto
    if hasattr(tf, "ConfigProto"):
        return tf.ConfigProto

    try:
        from tensorflow.core.protobuf import config_pb2

        return config_pb2.ConfigProto
    except ImportError as exc:
        raise ImportError(
            "TensorFlow ConfigProto is unavailable. Please install TensorFlow "
            "before configuring the MUSA graph optimizer."
        ) from exc


def _get_rewrite_options(config):
    return config.graph_options.rewrite_options


def _remove_custom_musa_graph_optimizer(rewrite_options):
    kept_custom_optimizers = [
        custom_optimizer
        for custom_optimizer in rewrite_options.custom_optimizers
        if custom_optimizer.name != MUSA_GRAPH_OPTIMIZER_NAME
    ]
    del rewrite_options.custom_optimizers[:]
    for custom_optimizer in kept_custom_optimizers:
        rewrite_options.custom_optimizers.add().CopyFrom(custom_optimizer)


def _remove_musa_graph_optimizer_from_optimizer_list(rewrite_options):
    kept_optimizers = [
        optimizer
        for optimizer in rewrite_options.optimizers
        if optimizer != MUSA_GRAPH_OPTIMIZER_NAME
    ]
    del rewrite_options.optimizers[:]
    rewrite_options.optimizers.extend(kept_optimizers)


def set_musa_graph_optimizer_enabled(
    config=None,
    enabled=True,
    add_to_optimizer_list=False,
):
    """Enable or disable the MUSA custom graph optimizer on a ConfigProto.

    Args:
        config: Optional `tf.compat.v1.ConfigProto` to update in place. When
            omitted, a new ConfigProto is created and returned.
        enabled: Whether to enable `musa_graph_optimizer`.
        add_to_optimizer_list: Also add the optimizer name to
            `rewrite_options.optimizers`. Most callers should leave this as
            False, which matches TensorFlow's custom optimizer registration
            style. Some tests use True to force an explicit Grappler pass list.

    Returns:
        The updated ConfigProto.
    """
    if config is None:
        config = _get_config_proto_class()()

    rewrite_options = _get_rewrite_options(config)
    _remove_custom_musa_graph_optimizer(rewrite_options)

    if enabled:
        custom_optimizer = rewrite_options.custom_optimizers.add()
        custom_optimizer.name = MUSA_GRAPH_OPTIMIZER_NAME
        if (
            add_to_optimizer_list
            and MUSA_GRAPH_OPTIMIZER_NAME not in rewrite_options.optimizers
        ):
            rewrite_options.optimizers.extend([MUSA_GRAPH_OPTIMIZER_NAME])
    else:
        _remove_musa_graph_optimizer_from_optimizer_list(rewrite_options)

    return config


def enable_musa_graph_optimizer(config=None, add_to_optimizer_list=False):
    """Enable `musa_graph_optimizer` on a ConfigProto."""
    return set_musa_graph_optimizer_enabled(
        config=config,
        enabled=True,
        add_to_optimizer_list=add_to_optimizer_list,
    )


def disable_musa_graph_optimizer(config=None):
    """Disable `musa_graph_optimizer` on a ConfigProto."""
    return set_musa_graph_optimizer_enabled(config=config, enabled=False)


def is_musa_graph_optimizer_enabled(config):
    """Return whether a ConfigProto enables `musa_graph_optimizer`."""
    rewrite_options = _get_rewrite_options(config)
    return any(
        custom_optimizer.name == MUSA_GRAPH_OPTIMIZER_NAME
        for custom_optimizer in rewrite_options.custom_optimizers
    )
