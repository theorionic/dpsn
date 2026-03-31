"""Export a trained DPSN checkpoint to .dpsn binary format.

.dpsn format layout
───────────────────
  [0:8]   Magic: b"DPSN\\x00\\x01\\x00\\x00"
  [8:16]  header_size: u64 LE  (byte length of the JSON header)
  [16:]   JSON header (UTF-8)
  [pad]   Zero-padding to next 64-byte boundary
  [...]   Sections, each 64-byte aligned

Dense tensor sections (controller / indexer / acc / retrieval_integrator / prefetch_*):
  For each tensor in the section:
    name_len : u16 LE
    name     : utf-8 bytes
    ndim     : u8
    shape    : ndim × u32 LE
    dtype    : u8  (0 = f32, 1 = bf16)
    data     : raw bytes (row-major)

Sparse pool section:
    rows     : u32 LE
    cols     : u32 LE
    d_model  : u32 LE
    n_entries: u32 LE
    dtype    : u8  (0 = f32, 1 = bf16)
    For each entry (sorted by row then col):
      row  : u16 LE
      col  : u16 LE
      vec  : d_model × dtype_bytes

Dense pool section (--no_prune):
    rows     : u32 LE
    cols     : u32 LE
    d_model  : u32 LE
    dtype    : u8
    data     : rows × cols × d_model × dtype_bytes  (row-major)

Coverage section:
    Raw UTF-8 JSON (same structure as pool_coverage_step_*.json)

Tokenizer section:
    Raw bytes of tokenizer.json (HuggingFace fast tokenizer format)

Usage
─────
  python export_dpsn.py \\
      --checkpoint_dir /kaggle/working/dpsn/checkpoints_xxl/ \\
      --output model.dpsn \\
      --config mini_pool \\
      --tokenizer EleutherAI/gpt-neo-125m

  # Dense pool (no pruning):
  python export_dpsn.py ... --no_prune

  # Specific coverage file:
  python export_dpsn.py ... --coverage /path/to/pool_coverage_step_500000.json
"""

import argparse
import glob
import io
import json
import os
import struct
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

MAGIC = b"DPSN\x00\x01\x00\x00"
ALIGN = 64  # all sections start on a 64-byte boundary
DTYPE_F32  = 0
DTYPE_BF16 = 1


# ── Alignment helpers ─────────────────────────────────────────────────────────

def _align_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) & ~(a - 1)


def _pad_to(f: io.RawIOBase, alignment: int = ALIGN) -> None:
    pos = f.tell()
    pad = _align_up(pos, alignment) - pos
    if pad:
        f.write(b"\x00" * pad)


# ── Tensor dtype helpers ──────────────────────────────────────────────────────

def _to_bf16_bytes(arr: np.ndarray) -> bytes:
    """Convert numpy array → bf16 bytes via the ml_dtypes package."""
    try:
        import ml_dtypes  # pip install ml_dtypes
        return arr.astype(ml_dtypes.bfloat16).tobytes()
    except ImportError:
        # Fallback: reinterpret float32 top-2 bytes as bf16 (big-endian trick).
        # Works correctly for non-NaN, non-inf values.
        f32 = arr.astype(np.float32)
        # bf16 = upper 16 bits of IEEE 754 float32 (same exponent, truncated mantissa)
        u32 = f32.view(np.uint32)
        bf16 = (u32 >> 16).astype(np.uint16)
        return bf16.tobytes()


def _best_dtype(arr: np.ndarray) -> tuple:
    """Return (dtype_byte, bytes) using bf16 if possible, f32 otherwise."""
    try:
        import ml_dtypes  # noqa: F401
        return DTYPE_BF16, _to_bf16_bytes(arr)
    except ImportError:
        # Check if ml_dtypes is available without import error on the conversion
        try:
            b = _to_bf16_bytes(arr)
            return DTYPE_BF16, b
        except Exception:
            return DTYPE_F32, arr.astype(np.float32).tobytes()


# ── Section serializers ───────────────────────────────────────────────────────

def _serialize_dense_tensors(tensors: dict) -> bytes:
    """Serialize a dict of {name: jax/numpy array} into the dense tensor format."""
    buf = io.BytesIO()
    for name, arr in tensors.items():
        np_arr = np.asarray(arr)
        dtype_byte, data = _best_dtype(np_arr)

        name_bytes = name.encode("utf-8")
        buf.write(struct.pack("<H", len(name_bytes)))
        buf.write(name_bytes)
        buf.write(struct.pack("<B", np_arr.ndim))
        for dim in np_arr.shape:
            buf.write(struct.pack("<I", dim))
        buf.write(struct.pack("<B", dtype_byte))
        buf.write(data)
    return buf.getvalue()


def _serialize_sparse_pool(pool_arr: np.ndarray,
                            accessed_coords: set) -> bytes:
    """Serialize pool as sparse entries (only trained coordinates)."""
    R, C, D = pool_arr.shape
    dtype_byte, _ = _best_dtype(np.zeros(1))  # probe dtype availability

    buf = io.BytesIO()
    buf.write(struct.pack("<IIII", R, C, D, len(accessed_coords)))
    buf.write(struct.pack("<B", dtype_byte))

    for (r, c) in sorted(accessed_coords):
        vec = pool_arr[r, c]  # shape (D,)
        _, vec_bytes = _best_dtype(vec)
        buf.write(struct.pack("<HH", r, c))
        buf.write(vec_bytes)

    return buf.getvalue()


def _serialize_dense_pool(pool_arr: np.ndarray) -> bytes:
    """Serialize the full pool as a dense [R, C, D] array."""
    R, C, D = pool_arr.shape
    dtype_byte, data = _best_dtype(pool_arr)

    buf = io.BytesIO()
    buf.write(struct.pack("<IIIB", R, C, D, dtype_byte))
    buf.write(data)
    return buf.getvalue()


def _load_tokenizer_bytes(tokenizer_name: str) -> bytes:
    """Download/load HuggingFace tokenizer and return tokenizer.json bytes."""
    import tempfile
    from transformers import AutoTokenizer

    print(f"  Loading tokenizer '{tokenizer_name}' ...")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    with tempfile.TemporaryDirectory() as tmp:
        tok.save_pretrained(tmp)
        tok_file = os.path.join(tmp, "tokenizer.json")
        if os.path.exists(tok_file):
            with open(tok_file, "rb") as f:
                data = f.read()
            print(f"  Tokenizer: {len(data) / 1024:.1f} KB")
            return data

    # Fallback: minimal vocab JSON
    print("  Warning: tokenizer.json not found, saving vocab only")
    return json.dumps({
        "model": tok.__class__.__name__,
        "vocab": tok.get_vocab(),
    }).encode("utf-8")


# ── Main writer ───────────────────────────────────────────────────────────────

def write_dpsn(
    output_path: str,
    params: dict,
    config_dict: dict,
    tokenizer_name: str,
    coverage_path: str | None = None,
    prune: bool = True,
) -> None:
    from flax import traverse_util

    print("\n[1/5] Flattening params ...")
    flat = traverse_util.flatten_dict(params, sep="/")

    # Split by top-level component name
    component_names = {
        "controller", "indexer", "pool", "acc",
        "retrieval_integrator", "prefetch_query_attn", "prefetch_query_proj",
    }
    components: dict[str, dict] = {c: {} for c in component_names}
    unknown = {}
    for key, val in flat.items():
        prefix = key.split("/")[0]
        if prefix in component_names:
            components[prefix][key] = val
        else:
            unknown[key] = val

    if unknown:
        print(f"  Warning: unrecognized param keys (skipped): {list(unknown)[:5]}")

    # Remove empty components
    components = {k: v for k, v in components.items() if v}
    print(f"  Components found: {list(components.keys())}")

    # Pool tensor
    pool_tensors = components.get("pool", {})
    pool_key = next((k for k in pool_tensors if "embedding" in k), None)
    if pool_key is None:
        # Fallback: pick the largest tensor
        pool_key = max(pool_tensors, key=lambda k: np.asarray(pool_tensors[k]).size, default=None)
    if pool_key is None:
        raise RuntimeError("No pool/embedding tensor found in checkpoint params")

    pool_arr = np.asarray(pool_tensors[pool_key])
    if pool_arr.ndim != 3:
        raise RuntimeError(f"Expected pool tensor shape (R, C, D), got {pool_arr.shape}")
    R, C, D = pool_arr.shape
    print(f"  Pool: {R}×{C}×{D} = {R*C*D:,} values  ({pool_arr.nbytes/1e9:.2f} GB f32)")

    print("\n[2/5] Loading coverage ...")
    accessed_coords = None
    coverage_raw = None
    if coverage_path and os.path.exists(coverage_path):
        with open(coverage_path) as f:
            cov_json = json.load(f)
        coverage_raw = cov_json.get("coverage_data", cov_json)
        freq = coverage_raw.get("access_frequency", {})
        accessed_coords = set()
        for key in freq:
            parts = key.split(",")
            accessed_coords.add((int(parts[0]), int(parts[1])))
        pct = len(accessed_coords) / (R * C) * 100
        print(f"  Trained coordinates: {len(accessed_coords):,} / {R*C:,}  ({pct:.1f}%)")
    else:
        print("  No coverage file — pool will be stored dense")

    print("\n[3/5] Building sections ...")
    sections_data: dict[str, bytes] = {}

    # tokenizer
    sections_data["tokenizer"] = _load_tokenizer_bytes(tokenizer_name)

    # dense component sections
    dense_components = [k for k in components if k != "pool"]
    for comp in dense_components:
        data = _serialize_dense_tensors(components[comp])
        sections_data[comp] = data
        print(f"  {comp}: {len(data) / 1e6:.1f} MB")

    # pool
    if prune and accessed_coords:
        data = _serialize_sparse_pool(pool_arr, accessed_coords)
        pool_format = "sparse_bf16"
        pool_stored = len(accessed_coords)
        print(f"  pool (sparse): {len(data) / 1e6:.1f} MB  "
              f"(saved {(1 - len(data) / (R*C*D*2)) * 100:.0f}% vs dense bf16)")
    else:
        data = _serialize_dense_pool(pool_arr)
        pool_format = "dense_bf16"
        pool_stored = R * C
        print(f"  pool (dense): {len(data) / 1e6:.1f} MB")
    sections_data["pool"] = data

    # coverage
    if coverage_raw:
        sections_data["coverage"] = json.dumps(coverage_raw).encode("utf-8")
        print(f"  coverage: {len(sections_data['coverage']) / 1e3:.1f} KB")

    print("\n[4/5] Computing layout ...")
    section_order = (
        ["tokenizer"]
        + [c for c in dense_components if c in sections_data]
        + ["pool"]
        + (["coverage"] if "coverage" in sections_data else [])
    )

    def _build_header(offsets: dict | None) -> bytes:
        """Build the JSON header. offsets=None for the first (sizing) pass."""
        section_list = []
        for name in section_order:
            info: dict = {"name": name, "size": len(sections_data[name])}
            if offsets:
                info["offset"] = offsets[name]
            if name == "tokenizer":
                info["format"] = "hf_tokenizer_json"
                info["hf_name"] = tokenizer_name
            elif name == "pool":
                info["format"] = pool_format
                info["shape"] = [R, C, D]
                info["stored_entries"] = pool_stored
            elif name == "coverage":
                info["format"] = "coord_freq_json"
            else:
                info["format"] = "dense_bf16"
                info["tensor_names"] = list(components[name].keys())
            section_list.append(info)

        hdr = {
            "version": 1,
            "model_type": "dpsn",
            "config": config_dict,
            "sections": section_list,
        }
        return json.dumps(hdr, indent=2).encode("utf-8")

    def _compute_offsets(header_json: bytes) -> dict:
        base = _align_up(8 + 8 + len(header_json))
        offsets = {}
        cur = base
        for name in section_order:
            offsets[name] = cur
            cur = _align_up(cur + len(sections_data[name]))
        return offsets

    # Two-pass layout (header size changes once offsets are known)
    header_json = _build_header(None)
    offsets = _compute_offsets(header_json)
    header_json = _build_header(offsets)
    offsets = _compute_offsets(header_json)
    header_json = _build_header(offsets)  # stable after 2nd pass

    total_bytes = max(offsets[name] + len(sections_data[name]) for name in section_order)
    print(f"  Total file size: {total_bytes / 1e9:.3f} GB")

    print(f"\n[5/5] Writing {output_path} ...")
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        _pad_to(f)

        for name in section_order:
            f.write(sections_data[name])
            _pad_to(f)

    actual = os.path.getsize(output_path)
    print(f"\nDone.  {output_path}  ({actual / 1e9:.3f} GB)")
    print("\nSection summary:")
    for name in section_order:
        print(f"  offset {offsets[name]:>12,}  size {len(sections_data[name]):>12,}  [{name}]")


# ── Checkpoint loading ────────────────────────────────────────────────────────

def _load_params(checkpoint_dir: str) -> tuple[dict, int]:
    """Load model params from Orbax checkpoint. Returns (params, step)."""
    import orbax.checkpoint as ocp

    abs_dir = os.path.abspath(checkpoint_dir)
    checkpointer = ocp.PyTreeCheckpointer()

    # Try CheckpointManager first
    try:
        mgr = ocp.CheckpointManager(abs_dir, checkpointer)
        step = mgr.latest_step()
        if step is not None:
            state = mgr.restore(step)
            params = state["params"] if isinstance(state, dict) and "params" in state else state.params
            print(f"  Loaded step {step} via CheckpointManager")
            return params, step
    except Exception as e:
        print(f"  CheckpointManager failed ({e}), scanning for step dirs ...")

    # Scan for step subdirectories
    step = None
    for item in os.listdir(abs_dir):
        if item.isdigit():
            candidate = int(item)
            if step is None or candidate > step:
                step = candidate

    if step is None:
        raise FileNotFoundError(f"No checkpoint steps found in {abs_dir}")

    for suffix in [f"{step}/default", str(step)]:
        path = os.path.join(abs_dir, suffix)
        if os.path.exists(path):
            try:
                state = checkpointer.restore(path)
                params = state["params"] if isinstance(state, dict) and "params" in state else state.params
                print(f"  Loaded step {step} from {path}")
                return params, step
            except Exception as e:
                print(f"  Restore from {path} failed: {e}")

    raise FileNotFoundError(f"Could not restore checkpoint from {abs_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DPSN Orbax checkpoint → .dpsn binary format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to Orbax checkpoint directory")
    parser.add_argument("--output", required=True,
                        help="Output .dpsn file path")
    parser.add_argument("--config", default="mini_pool",
                        help="Model config name (default: mini_pool)")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neo-125m",
                        help="HuggingFace tokenizer name")
    parser.add_argument("--coverage", default=None,
                        help="Path to pool_coverage_step_*.json. "
                             "Auto-detected from checkpoint_dir if not set.")
    parser.add_argument("--no_prune", action="store_true",
                        help="Store full dense pool even if coverage is available")
    args = parser.parse_args()

    print("=== DPSN Checkpoint Exporter ===")
    print(f"  checkpoint : {args.checkpoint_dir}")
    print(f"  output     : {args.output}")
    print(f"  config     : {args.config}")
    print(f"  tokenizer  : {args.tokenizer}")
    print(f"  prune pool : {not args.no_prune}")

    # Load config
    from dpsn_r_jax.config import get_model_config
    cfg = get_model_config(args.config)
    # Build a plain dict from the dataclass (skip None values for cleanliness)
    config_dict = {k: v for k, v in asdict(cfg).items() if v is not None}

    # Load params
    print("\n[1/5] Loading checkpoint ...")
    params, step = _load_params(args.checkpoint_dir)
    print(f"  Checkpoint step: {step}")

    # Auto-detect coverage
    coverage_path = args.coverage
    if not coverage_path:
        pattern = os.path.join(args.checkpoint_dir, "pool_coverage_step_*.json")
        cov_files = sorted(glob.glob(pattern))
        if cov_files:
            coverage_path = cov_files[-1]
            print(f"  Auto-detected coverage: {Path(coverage_path).name}")
        else:
            print("  No coverage file found — pool stored dense")

    write_dpsn(
        output_path=args.output,
        params=params,
        config_dict=config_dict,
        tokenizer_name=args.tokenizer,
        coverage_path=coverage_path,
        prune=not args.no_prune,
    )


if __name__ == "__main__":
    main()
