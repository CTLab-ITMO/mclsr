# use
# python scripts/generate_item_counts.py     --input ./data/Clothing/train_sasrec.txt     --output ./data/Clothing/item_counts.pkl     --num_items 23033

import pickle
import numpy as np
from collections import Counter
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Generate item interaction counts for LogQ correction.")
    parser.add_argument("--input", type=str, required=True, help="Path to train_sasrec.txt")
    parser.add_argument("--output", type=str, required=True, help="Path to save item_counts.pkl")
    parser.add_argument("--num_items", type=int, required=True, help="Number of items in dataset (from dataset.meta)")
    
    args = parser.parse_args()

    # We use num_items + 2 because the embedding layer size is num_items + 2 
    # (reserved for padding at index 0 and mask at index num_items + 1)
    array_size = args.num_items + 2
    counts = Counter()

    print(f"[*] Reading dataset from: {args.input}")
    if not os.path.exists(args.input):
        print(f"[!] Error: File {args.input} not found.")
        return

    with open(args.input, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            # parts[0] is user_id, parts[1:] are sequences of interacted item_ids
            items = [int(i) for i in parts[1:]]
            counts.update(items)

    # Initialize frequencies array with zeros
    item_counts_array = np.zeros(array_size, dtype=np.float32)
    
    for item_id, count in counts.items():
        if item_id < array_size:
            item_counts_array[item_id] = count
        else:
            print(f"[!] Warning: item_id {item_id} exceeds array size {array_size}. Check your num_items!")

    # Numerical stability: set zero counts to 1.0 to avoid log(0) in LogQ correction
    zero_mask = (item_counts_array == 0)
    num_zeros = np.sum(zero_mask)
    if num_zeros > 0:
        print(f"[*] Found {num_zeros} items with zero interactions. Setting their count to 1.0 for stability.")
        item_counts_array[zero_mask] = 1.0

    print(f"[*] Saving popularity statistics to: {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(item_counts_array, f)
    
    print("[+] Done! LogQ data is ready.")

if __name__ == "__main__":
    main()
