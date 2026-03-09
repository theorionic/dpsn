import argparse
import os
import time
import numpy as np
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial

# Import your tokenizer mechanism
from dpsn_r_jax.data.tokenizer import get_tokenizer

def process_chunk(texts, tokenizer_name, max_seq_length):
    """
    Process a list of texts into tokenized integer arrays.
    This runs in a separate worker process.
    """
    # Re-initialize tokenizer in worker process
    tokenizer = get_tokenizer(tokenizer_name)
    
    # Check if HF tokenizer
    is_hf = hasattr(tokenizer, "__call__") and not hasattr(tokenizer, "max_val")
    
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_id is None:
        pad_id = 0
        
    chunk_ids = []
    
    if is_hf:
        # Batch encode with HF Tokenizer
        encoded = tokenizer(
            texts,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return encoded["input_ids"].astype(np.int32)
    else:
        # Basic tokenizer encode loop
        for text in texts:
            ids = tokenizer.encode(text)
            if len(ids) > max_seq_length:
                ids = ids[:max_seq_length]
            else:
                ids = ids + [pad_id] * (max_seq_length - len(ids))
            chunk_ids.append(ids)
            
        return np.array(chunk_ids, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize HF Dataset for Google Grain")
    parser.add_argument("--hf_dataset", type=str, required=True, help="dataset name (e.g. openbmb/Ultra-FineWeb)")
    parser.add_argument("--hf_subset", type=str, default=None, help="Subset name if any")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--text_column", type=str, default="text", help="Text column name")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer name (e.g. EleutherAI/gpt-neo-125M)")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .npy chunks")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Number of rows per .npy file")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process in total")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of CPU cores to use")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    workers = args.num_workers if args.num_workers else max(1, cpu_count() - 1)
    print(f"Starting pre-tokenization with {workers} parallel workers...")
    
    print(f"Loading dataset: {args.hf_dataset} (streaming=True)")
    dataset = load_dataset(args.hf_dataset, name=args.hf_subset, split=args.split, streaming=True)
    
    total_processed = 0
    chunk_index = 0
    current_texts = []
    
    start_time = time.time()
    
    with Pool(processes=workers) as pool:
        process_func = partial(process_chunk, tokenizer_name=args.tokenizer, max_seq_length=args.seq_len)
        
        # Batching sizes for multiprocessing map
        # We accumulate chunks of text, then farm them out to workers in smaller batches
        MAPPING_BATCH_SIZE = 5000 
        
        for item in dataset:
            text = item.get(args.text_column) or item.get("content") or item.get("sentence") or ""
            if not text:
                continue
                
            current_texts.append(text)
            
            if len(current_texts) >= args.chunk_size:
                print(f"Processing chunk {chunk_index} ({args.chunk_size} samples)...")
                
                # Split current_texts into smaller batches for pool.map
                sub_batches = [current_texts[i:i + MAPPING_BATCH_SIZE] for i in range(0, len(current_texts), MAPPING_BATCH_SIZE)]
                
                # Map across processes
                results = pool.map(process_func, sub_batches)
                
                # Concatenate the resulting numpy arrays
                final_array = np.concatenate(results, axis=0)
                
                # Save to disk
                out_path = os.path.join(args.output_dir, f"chunk_{chunk_index:05d}.npy")
                np.save(out_path, final_array)
                print(f"Saved {out_path} shape={final_array.shape}")
                
                total_processed += len(current_texts)
                chunk_index += 1
                current_texts = []
                
                if args.max_samples and total_processed >= args.max_samples:
                    print(f"Reached max_samples {args.max_samples}. Stopping.")
                    break
                    
        # Process remainder
        if current_texts and (not args.max_samples or total_processed < args.max_samples):
            print(f"Processing final chunk {chunk_index} ({len(current_texts)} samples)...")
            sub_batches = [current_texts[i:i + MAPPING_BATCH_SIZE] for i in range(0, len(current_texts), MAPPING_BATCH_SIZE)]
            results = pool.map(process_func, sub_batches)
            final_array = np.concatenate(results, axis=0)
            
            out_path = os.path.join(args.output_dir, f"chunk_{chunk_index:05d}.npy")
            np.save(out_path, final_array)
            print(f"Saved {out_path} shape={final_array.shape}")
            total_processed += len(current_texts)
            
    elapsed = time.time() - start_time
    print(f"Successfully processed {total_processed} samples in {elapsed:.2f} seconds.")
    print(f"Your pre-tokenized data is ready in: {args.output_dir}")
    print(f"Update your main.py to read these files using Grain (configure grain_loader.py to read .npy).")

if __name__ == "__main__":
    main()
