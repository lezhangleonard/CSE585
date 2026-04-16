#TODO in progress
import json
import os
from pathlib import Path

def extract_correctness_stats(base_path, run_type, batch_ratios):
    """
    Traverses the directory structure and extracts 'correctness' from stats.json.
    """
    results = []
    # Convert base_path to a Path object for easier joining
    root = Path(base_path) / "eval_runs" / run_type
    
    # The subfolders we expect inside each workload
    modes = ["batch", "dag", "sequential"]

    if not root.exists():
        print(f"Error: Base directory {root} does not exist.")
        return results

    for br in batch_ratios:
        br_folder = root / f"br_{br}"
        
        if not br_folder.exists():
            print(f"Skipping: {br_folder} not found.")
            continue

        # Iterate through every workload folder inside br_X
        for workload_dir in br_folder.iterdir():
            if workload_dir.is_dir():
                workload_name = workload_dir.name
                
                if "w_500_" in workload_name:
                    continue

                for mode in modes:
                    stats_path = workload_dir / mode / "stats.json"
                    
                    if stats_path.exists():
                        try:
                            with open(stats_path, 'r') as f:
                                data = json.load(f)
                                # Using .get() prevents the script from crashing if 'correctness' is missing
                                correctness = data.get("correctness", "KEY_NOT_FOUND")
                                
                                results.append({
                                    "run_type": run_type,
                                    "batch_ratio": br,
                                    "workload": workload_name,
                                    "mode": mode,
                                    "correctness": correctness
                                })
                        except json.JSONDecodeError:
                            print(f"Error: Failed to decode JSON in {stats_path}")
                        except Exception as e:
                            print(f"Error: Could not read {stats_path}: {e}")
                            
    return results

def main():
    # --- Configuration ---
    BASE_DIRECTORY = "/scratch/engin_root/engin1/arshiv/ml/agentic_kg"  # Change this to the path where 'final_runs' is located
    RUN_TYPE = "real"
    BATCH_RATIOS = [4, 8, 16]

    print(f"Extracting stats for run_type='{RUN_TYPE}'...")
    
    # --- Execution ---
    data = extract_correctness_stats(BASE_DIRECTORY, RUN_TYPE, BATCH_RATIOS)

    # --- Output ---
    if not data:
        print("No data found. Check your directory paths.")
    else:
        # Print header
        print(f"{'BR':<5} | {'Workload':<20} | {'Mode':<12} | {'Correctness'}")
        print("-" * 60)
        
        for entry in data:
            print(f"{entry['batch_ratio']:<5} | "
                  f"{entry['workload']:<20} | "
                  f"{entry['mode']:<12} | "
                  f"{entry['correctness']}")

if __name__ == "__main__":
    main()