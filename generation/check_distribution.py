import pandas as pd
import os
import argparse

def analyze_has_person_distribution(file_path):
    """
    Analyze the distribution of 'has_person' labels in a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        dict: Distribution statistics
    """
    print(f"Analyzing file: {file_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if 'has_person' column exists
        if 'has_person' not in df.columns:
            print(f"Error: 'has_person' column not found in {file_path}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Calculate distribution
        has_person_counts = df['has_person'].value_counts().sort_index()
        total_samples = len(df)
        
        # Create detailed statistics
        stats = {
            'file': os.path.basename(file_path),
            'total_samples': total_samples,
            'has_person_0': has_person_counts.get(0, 0),
            'has_person_1': has_person_counts.get(1, 0),
            'has_person_0_percent': (has_person_counts.get(0, 0) / total_samples) * 100,
            'has_person_1_percent': (has_person_counts.get(1, 0) / total_samples) * 100
        }
        
        return stats
        
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def print_distribution_table(stats_list):
    """Print a formatted table of distribution statistics."""
    print("\n" + "="*80)
    print("HAS_PERSON DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"{'File':<25} {'Total':<10} {'has_person=0':<15} {'has_person=1':<15} {'Ratio (0:1)':<12}")
    print("-"*80)
    
    overall_stats = {'total': 0, 'has_person_0': 0, 'has_person_1': 0}
    
    for stats in stats_list:
        if stats is None:
            continue
            
        ratio = f"{stats['has_person_0']}:{stats['has_person_1']}"
        print(f"{stats['file']:<25} {stats['total_samples']:<10} "
              f"{stats['has_person_0']:<6}({stats['has_person_0_percent']:.1f}%) "
              f"{stats['has_person_1']:<6}({stats['has_person_1_percent']:.1f}%) "
              f"{ratio:<12}")
        
        # Accumulate overall statistics
        overall_stats['total'] += stats['total_samples']
        overall_stats['has_person_0'] += stats['has_person_0']
        overall_stats['has_person_1'] += stats['has_person_1']
    
    # Print overall statistics
    if overall_stats['total'] > 0:
        overall_0_percent = (overall_stats['has_person_0'] / overall_stats['total']) * 100
        overall_1_percent = (overall_stats['has_person_1'] / overall_stats['total']) * 100
        overall_ratio = f"{overall_stats['has_person_0']}:{overall_stats['has_person_1']}"
        
        print("-"*80)
        print(f"{'OVERALL':<25} {overall_stats['total']:<10} "
              f"{overall_stats['has_person_0']:<6}({overall_0_percent:.1f}%) "
              f"{overall_stats['has_person_1']:<6}({overall_1_percent:.1f}%) "
              f"{overall_ratio:<12}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Analyze has_person distribution in ECHR datasets')
    parser.add_argument('--dataset_dir', type=str, default='dataset', 
                       help='Directory containing the CSV files (default: dataset)')
    parser.add_argument('--files', nargs='+', 
                       help='Specific files to analyze (default: all echr_*.csv files)')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed breakdown by file')
    
    args = parser.parse_args()
    
    # Determine which files to analyze
    if args.files:
        files_to_analyze = [os.path.join(args.dataset_dir, f) if not os.path.isabs(f) else f 
                           for f in args.files]
    else:
        # Default: analyze all ECHR CSV files in the dataset directory
        dataset_dir = args.dataset_dir
        if not os.path.exists(dataset_dir):
            print(f"Error: Dataset directory '{dataset_dir}' not found")
            return
        
        csv_files = [f for f in os.listdir(dataset_dir) 
                    if f.startswith('echr_') and f.endswith('.csv') and not f.endswith('.backup')]
        files_to_analyze = [os.path.join(dataset_dir, f) for f in sorted(csv_files)]
    
    if not files_to_analyze:
        print("No CSV files found to analyze")
        return
    
    print(f"Found {len(files_to_analyze)} files to analyze:")
    for f in files_to_analyze:
        print(f"  - {f}")
    
    # Analyze each file
    all_stats = []
    for file_path in files_to_analyze:
        if os.path.exists(file_path):
            stats = analyze_has_person_distribution(file_path)
            all_stats.append(stats)
        else:
            print(f"Warning: File not found: {file_path}")
    
    # Print results
    if args.detailed:
        for stats in all_stats:
            if stats:
                print(f"\nDetailed analysis for {stats['file']}:")
                print(f"  Total samples: {stats['total_samples']:,}")
                print(f"  has_person = 0: {stats['has_person_0']:,} ({stats['has_person_0_percent']:.2f}%)")
                print(f"  has_person = 1: {stats['has_person_1']:,} ({stats['has_person_1_percent']:.2f}%)")
    
    # Print summary table
    print_distribution_table(all_stats)

if __name__ == "__main__":
    main()