import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from dpsn_r_jax.config import get_model_config
from dpsn_r_jax.models.dpsnr import DPSNR
import numpy as np


def check_distribution():
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"Device Count: {jax.device_count()}")
    print(f"Local Devices: {jax.local_devices()}")

    # 1. Setup Mesh (Same logic as main.py)
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names=("shard",))
    print(f"\nMesh Created: {mesh}")

    # 2. Define Sharding Rules
    pool_sharding = NamedSharding(mesh, PartitionSpec("shard", None))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    def get_sharding_rule(path, param):
        if "pool" in path:
            return pool_sharding
        return replicated_sharding

    # 3. Initialize Model (Tiny for quick check)
    config = get_model_config("tiny")
    model = DPSNR(config)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)

    print("\nInitializing model...")

    # Abstract initialization to check shapes/sharding without allocating memory yet
    @jax.jit
    def init_model(rng, input_ids):
        return model.init(rng, input_ids)

    abstract_variables = jax.eval_shape(init_model, rng, dummy_input)

    # Create expected sharding tree
    sharding_tree = jax.tree_util.tree_map_with_path(
        get_sharding_rule, abstract_variables
    )

    # Actually initialize with sharding constraints
    variables = jax.lax.with_sharding_constraint(
        init_model(rng, dummy_input), sharding_tree
    )

    params = variables["params"]

    # 4. Verify Sharding
    print("\n--- SHARDING VERIFICATION ---")

    # Check Pool (Should be sharded)
    # Note: 'params' might differ if your model structure changed, checking specific known keys
    if "pool" in params:
        pool_params = params["pool"]
        # Assuming there is a 'vectors' or similar large parameter in pool
        # We'll just look at the first leaf in the pool subtree
        pool_leaf = jax.tree_util.tree_leaves(pool_params)[0]
        print(f"\n[Pool Params] (Expected: Sharded along 'shard')")
        print(f"Shape: {pool_leaf.shape}")
        print(f"Sharding: {pool_leaf.sharding}")

        is_fully_replicated = pool_leaf.sharding.is_fully_replicated
        print(f"Is Fully Replicated? {is_fully_replicated}")
        if not is_fully_replicated:
            print("✅ SUCCESS: Pool params are distributed!")
        else:
            if jax.device_count() > 1:
                print(
                    "❌ WARNING: Pool params are replicated despite multiple devices."
                )
            else:
                print("ℹ️ Note: On single device, sharding == replication.")

    # Check Controller (Should be replicated)
    if "controller" in params:
        ctrl_params = params["controller"]
        ctrl_leaf = jax.tree_util.tree_leaves(ctrl_params)[0]
        print(f"\n[Controller Params] (Expected: Replicated)")
        print(f"Shape: {ctrl_leaf.shape}")
        print(f"Sharding: {ctrl_leaf.sharding}")

        # Replicated sharding often shows as NamedSharding(mesh, PartitionSpec())
        # or equivalent
        print(f"Is Fully Replicated? {ctrl_leaf.sharding.is_fully_replicated}")
        if ctrl_leaf.sharding.is_fully_replicated:
            print("✅ SUCCESS: Controller params are replicated on all devices.")


if __name__ == "__main__":
    check_distribution()
