from pathlib import Path
import argparse
import sys
import pandas as pd


def print_matches(path: Path):
    print(f"\n--- Checking file: {path} ---")
    df = pd.read_csv(path)

    cols = {c.lower(): c for c in df.columns}

    pred_col = cols['predicted']
    true_col = cols['true']

    matches = df[df[pred_col] == df[true_col]]

    if matches.empty:
        print("No matches (no rows where Predicted == True).")
        return

    print(f"Found {len(matches)} matching rows:\n")
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(matches.to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', help='CSV files to check',
                        default=[
                            '/home/dpereira/CB-LLMs/disentangling/work/check.csv',
                            '/home/dpereira/CB-LLMs/disentangling/work/check_vib.csv'
                        ])
    args = parser.parse_args(argv)

    paths = [Path(f) for f in args.files]
    for p in paths:
        print_matches(p)

    # Print rows from check_vib.csv where check.csv has true=pred
    print("\n" + "="*80)
    print("Rows from check_vib.csv where check.csv has Predicted == True:")
    print("="*80)
    
    check_path = Path('/home/dpereira/CB-LLMs/disentangling/work/check.csv')
    check_vib_path = Path('/home/dpereira/CB-LLMs/disentangling/work/check_vib.csv')
    
    check_df = pd.read_csv(check_path)
    check_vib_df = pd.read_csv(check_vib_path)
    
    # Find rows where check.csv has Predicted == True
    matching_indices = check_df[check_df['Predicted'] == check_df['True']].index
    
    # Get corresponding rows from check_vib.csv
    vib_matching_rows = check_vib_df.iloc[matching_indices]
    
    print(f"\nFound {len(vib_matching_rows)} rows where check.csv has Predicted == True:\n")
    with pd.option_context('display.max_columns', None, 'display.width', 200, 'display.max_colwidth', None):
        print(vib_matching_rows.to_string(index=True))
    

if __name__ == '__main__':
    main()
