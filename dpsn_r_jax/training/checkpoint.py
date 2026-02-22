"""Unified checkpoint handling with Orbax + TPU/multi-device support.

This module provides a single source of truth for checkpoint operations,
supporting both single-device and multi-device (TPU/GPU mesh) training.
"""

import os
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import sharding
from flax.training import train_state
import orbax.checkpoint as ocp
from absl import logging


def get_mesh(
    mesh_shape: Optional[Tuple[int, ...]] = None,
    axis_names: Optional[Tuple[str, ...]] = None,
) -> Optional[jax.sharding.Mesh]:
    """Create a mesh for multi-device training.

    Args:
        mesh_shape: Shape of the mesh. If None, creates 1D mesh with all devices.
        axis_names: Names for each mesh axis. Defaults to ('data',).

    Returns:
        Mesh object if multiple devices, None otherwise.
    """
    devices = jax.devices()
    num_devices = len(devices)

    if num_devices == 1:
        logging.info("Single device detected, no mesh created.")
        return None

    if mesh_shape is None:
        mesh_shape = (num_devices,)

    if axis_names is None:
        axis_names = ("data",)

    if len(mesh_shape) != len(axis_names):
        raise ValueError(
            f"mesh_shape length ({len(mesh_shape)}) must match "
            f"axis_names length ({len(axis_names)})"
        )

    device_array = np.array(devices).reshape(mesh_shape)
    mesh = jax.sharding.Mesh(device_array, axis_names)

    logging.info(f"Created mesh with shape {mesh_shape} and axes {axis_names}")
    return mesh


def get_sharding_spec(
    mesh: Optional[jax.sharding.Mesh],
    spec: Optional[Union[sharding.PartitionSpec, Tuple[str, ...]]] = None,
) -> Optional[jax.sharding.NamedSharding]:
    """Get NamedSharding for a given mesh and partition spec.

    Args:
        mesh: Device mesh. If None, returns None (no sharding).
        spec: Partition spec for sharding. If None, uses replicated sharding.

    Returns:
        NamedSharding object if mesh is provided, None otherwise.
    """
    if mesh is None:
        return None

    if spec is None:
        spec = sharding.PartitionSpec()

    if isinstance(spec, tuple):
        spec = sharding.PartitionSpec(*spec)

    return jax.sharding.NamedSharding(mesh, spec)


def create_checkpoint_manager(
    checkpoint_dir: str,
    max_to_keep: int = 3,
    create: bool = True,
) -> ocp.CheckpointManager:
    """Create an Orbax checkpoint manager.

    Args:
        checkpoint_dir: Directory to store checkpoints.
        max_to_keep: Maximum number of checkpoints to keep.
        create: Whether to create the directory if it doesn't exist.

    Returns:
        CheckpointManager instance.
    """
    abs_dir = os.path.abspath(checkpoint_dir)

    if create:
        os.makedirs(abs_dir, exist_ok=True)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        create=create,
    )

    return ocp.CheckpointManager(abs_dir, ocp.PyTreeCheckpointer(), options)


def save_checkpoint(
    checkpoint_dir: str,
    state: train_state.TrainState,
    step: int,
    max_to_keep: int = 3,
) -> None:
    """Save a training state checkpoint.

    Args:
        checkpoint_dir: Directory to save the checkpoint.
        state: Training state to save.
        step: Current training step.
        max_to_keep: Maximum number of checkpoints to keep.
    """
    mgr = create_checkpoint_manager(checkpoint_dir, max_to_keep=max_to_keep)
    mgr.save(step, state)
    mgr.wait_until_finished()

    logging.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")


def load_checkpoint(
    checkpoint_dir: str,
    target_state: Optional[train_state.TrainState] = None,
    step: Optional[int] = None,
) -> Tuple[train_state.TrainState, int]:
    """Load a checkpoint from directory.

    This function tries multiple checkpoint formats:
    1. Orbax CheckpointManager (recommended for TPU)
    2. Direct Orbax checkpoint path
    3. Legacy JSON format (for backward compatibility)

    Args:
        checkpoint_dir: Directory containing the checkpoint.
        target_state: Target state for sharding information. Required for
            multi-device loading to determine correct shard placement.
        step: Specific step to load. If None, loads the latest.

    Returns:
        Tuple of (loaded_state, step).

    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    abs_dir = os.path.abspath(checkpoint_dir)

    if not os.path.exists(abs_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {abs_dir}")

    checkpointer = ocp.PyTreeCheckpointer()

    # Try 1: Orbax CheckpointManager format
    try:
        mgr = ocp.CheckpointManager(abs_dir, checkpointer)

        if step is None:
            step = mgr.latest_step()

        if step is not None:
            if target_state is not None:
                # Use target state for sharding information
                state = mgr.restore(step, items=target_state)
            else:
                state = mgr.restore(step)

            logging.info(f"Loaded checkpoint from step {step} (Orbax Manager)")
            return state, step

    except Exception as e:
        logging.warning(f"Orbax CheckpointManager failed: {e}")

    # Try 2: Direct Orbax checkpoint path
    # First, discover step directories if step not provided or not found
    if step is None:
        # Scan for step directories (directories named with integers)
        try:
            for item in os.listdir(abs_dir):
                if item.isdigit():
                    item_path = os.path.join(abs_dir, item)
                    if os.path.isdir(item_path):
                        # Check if it has Orbax checkpoint structure
                        if os.path.exists(os.path.join(item_path, "default")):
                            if step is None or int(item) > step:
                                step = int(item)
        except Exception:
            pass

    potential_paths = []
    if step is not None:
        # Primary path: step/default (Orbax format)
        potential_paths.append(os.path.join(abs_dir, str(step), "default"))
        # Alternative: step directory itself
        potential_paths.append(os.path.join(abs_dir, str(step)))
        potential_paths.append(os.path.join(abs_dir, f"checkpoint_{step}"))

    # Also check for checkpoint at root level
    for subdir in ["default", "latest"]:
        potential_paths.append(os.path.join(abs_dir, subdir))

    for path in potential_paths:
        if os.path.exists(path):
            try:
                if target_state is not None:
                    state = checkpointer.restore(path, items=target_state)
                else:
                    state = checkpointer.restore(path)

                loaded_step = _extract_step_from_path(path, step)
                logging.info(f"Loaded checkpoint from {path} (Orbax Direct)")
                return state, loaded_step

            except Exception as e:
                logging.debug(f"Direct path restore failed: {e}")

    # Try 3: Legacy JSON format (backward compatibility)
    state, loaded_step = _try_load_legacy_json(abs_dir, target_state, step)
    if state is not None:
        logging.info(f"Loaded checkpoint from step {loaded_step} (Legacy JSON)")
        return state, loaded_step

    raise FileNotFoundError(f"No valid checkpoint found in {abs_dir}")


def load_pretrained_checkpoint(
    pretrained_path: str,
    target_state: train_state.TrainState,
) -> train_state.TrainState:
    """Load a pretrained checkpoint for fine-tuning.

    This is a convenience wrapper around load_checkpoint that loads only
    model parameters (not optimizer state), which is typical for transfer
    learning / fine-tuning scenarios.

    Uses partial restore to handle structure mismatches between saved
    checkpoint and target state (e.g., different optimizer state format).

    Args:
        pretrained_path: Path to pretrained checkpoint.
        target_state: Target state with correct model architecture and sharding.

    Returns:
        Training state with pretrained parameters loaded.
    """
    abs_dir = os.path.abspath(pretrained_path)

    if not os.path.exists(abs_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {abs_dir}")

    checkpointer = ocp.PyTreeCheckpointer()

    # Discover step directory
    step = None
    try:
        mgr = ocp.CheckpointManager(abs_dir, checkpointer)
        step = mgr.latest_step()
    except Exception:
        pass

    if step is None:
        # Scan for step directories
        try:
            for item in os.listdir(abs_dir):
                if item.isdigit():
                    item_path = os.path.join(abs_dir, item)
                    if os.path.isdir(item_path) and os.path.exists(
                        os.path.join(item_path, "default")
                    ):
                        if step is None or int(item) > step:
                            step = int(item)
        except Exception:
            pass

    if step is None:
        raise FileNotFoundError(f"No checkpoint step found in {abs_dir}")

    # Build path to checkpoint
    ckpt_path = os.path.join(abs_dir, str(step), "default")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(abs_dir, str(step))

    logging.info(f"Loading pretrained weights from {ckpt_path}")

    # Create a partial target with only params for partial restore
    # This handles structure mismatches (e.g., opt_state dict vs list)
    params_target = {"params": target_state.params}

    # Build restore args for partial restore
    restore_args = ocp.checkpoint_utils.construct_restore_args(
        params_target,
        jax.tree_util.tree_map(lambda x: x.sharding, params_target),
    )

    try:
        # Try partial restore - only loads params, ignores other state
        restored = checkpointer.restore(
            ckpt_path,
            items=params_target,
            restore_args=restore_args,
        )
        params = restored["params"]
    except Exception as e:
        logging.warning(f"Partial restore failed: {e}, trying full restore")
        # Fallback: try full restore and extract params
        try:
            restored = checkpointer.restore(ckpt_path)
            params = restored["params"]
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load pretrained checkpoint: {e2}"
            ) from e2

    # Truncate position embeddings if needed (fine-tuning with shorter seq length)
    params = _truncate_position_embeddings(params, target_state.params)

    # Create new state with loaded params but fresh optimizer
    state = target_state.replace(params=params)

    logging.info(f"Loaded pretrained params from step {step}")
    return state


def _truncate_position_embeddings(
    loaded_params: dict,
    target_params: dict,
) -> dict:
    """Truncate position embeddings if checkpoint has longer max_seq_len.

    When fine-tuning with a shorter max_seq_length than pretraining,
    we need to slice the position embeddings to match the target shape.

    Args:
        loaded_params: Parameters loaded from checkpoint.
        target_params: Target parameters (defines expected shapes).

    Returns:
        Parameters with truncated position embeddings if needed.
    """
    # Find position embedding paths (can be nested in modules)
    def find_pos_embed_keys(d: dict, prefix: str = "") -> list:
        keys = []
        for k, v in d.items():
            full_key = f"{prefix}/{k}" if prefix else k
            if k == "pos_encoding" and isinstance(v, dict) and "embedding" in v:
                keys.append((full_key + "/embedding", "pos_encoding", "embedding"))
            elif isinstance(v, dict):
                keys.extend(find_pos_embed_keys(v, full_key))
        return keys

    pos_embed_keys = find_pos_embed_keys(loaded_params)

    for full_path, module_key, embed_key in pos_embed_keys:
        try:
            # Navigate to the embedding in loaded params
            loaded_embed = loaded_params[module_key][embed_key]
            target_embed = target_params[module_key][embed_key]

            loaded_shape = loaded_embed.shape
            target_shape = target_embed.shape

            if loaded_shape != target_shape:
                # Only truncate if loaded is larger (can't extend)
                if loaded_shape[0] > target_shape[0]:
                    logging.info(
                        f"Truncating {full_path} from {loaded_shape} to {target_shape}"
                    )
                    loaded_params[module_key][embed_key] = loaded_embed[: target_shape[0]]
                elif loaded_shape[0] < target_shape[0]:
                    logging.warning(
                        f"Cannot extend {full_path} from {loaded_shape} to {target_shape}. "
                        f"Pretrained model has shorter max_seq_len. Using as-is."
                    )
        except (KeyError, AttributeError) as e:
            logging.debug(f"Could not process {full_path}: {e}")
            continue

    return loaded_params


def _extract_step_from_path(path: str, default_step: Optional[int]) -> int:
    """Extract step number from checkpoint path."""
    if default_step is not None:
        return default_step

    # Try to extract step from path
    parts = path.split(os.sep)
    for part in reversed(parts):
        if part.isdigit():
            return int(part)

    return 0


def _try_load_legacy_json(
    checkpoint_dir: str,
    target_state: Optional[train_state.TrainState],
    step: Optional[int],
) -> Tuple[Optional[train_state.TrainState], Optional[int]]:
    """Try to load legacy JSON checkpoint format.

    This provides backward compatibility with checkpoints saved in the
    previous JSON-based format.
    """
    import json

    # Look for params.json
    params_file = os.path.join(checkpoint_dir, "params.json")
    if os.path.exists(params_file):
        try:
            with open(params_file, "r") as f:
                params_dict = json.load(f)

            params = jax.tree_util.tree_map(jnp.array, params_dict)

            if target_state is not None:
                state = target_state.replace(params=params)
            else:
                raise ValueError("target_state required for legacy JSON loading")

            loaded_step = step if step is not None else 0
            return state, loaded_step

        except Exception as e:
            logging.debug(f"Legacy JSON loading failed: {e}")

    # Look for step-specific files
    if step is not None:
        step_params = os.path.join(checkpoint_dir, f"params_step_{step}.json")
        if os.path.exists(step_params):
            try:
                with open(step_params, "r") as f:
                    params_dict = json.load(f)

                params = jax.tree_util.tree_map(jnp.array, params_dict)

                checkpoint_file = os.path.join(
                    checkpoint_dir, f"checkpoint_{step}.json"
                )
                if os.path.exists(checkpoint_file) and target_state is not None:
                    with open(checkpoint_file, "r") as f:
                        ckpt_data = json.load(f)

                    # Restore pool momenta if present
                    if hasattr(target_state, "pool_m"):
                        target_state = target_state.replace(
                            params=params,
                            pool_m=jnp.array(ckpt_data.get("pool_m", [])),
                            pool_v=jnp.array(ckpt_data.get("pool_v", [])),
                            step=ckpt_data.get("step", step),
                        )
                    else:
                        target_state = target_state.replace(params=params)

                return target_state, step

            except Exception as e:
                logging.debug(f"Step-specific JSON loading failed: {e}")

    return None, None


def get_latest_step(checkpoint_dir: str) -> Optional[int]:
    """Get the latest checkpoint step without loading the checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        Latest step number, or None if no checkpoints found.
    """
    abs_dir = os.path.abspath(checkpoint_dir)

    if not os.path.exists(abs_dir):
        return None

    try:
        mgr = ocp.CheckpointManager(abs_dir, ocp.PyTreeCheckpointer())
        return mgr.latest_step()
    except Exception:
        pass

    # Fallback: scan directory for step numbers
    max_step = None
    for item in os.listdir(abs_dir):
        if item.isdigit():
            step = int(item)
            if max_step is None or step > max_step:
                max_step = step

    return max_step
