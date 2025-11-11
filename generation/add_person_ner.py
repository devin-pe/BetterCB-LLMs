#!/usr/bin/env python3
"""
add_person_ner.py

Add PERSON entity detection to ECHR CSV files (echr_train.csv, echr_test.csv, echr_validation.csv).
Uses Flair NER (flair/ner-english-ontonotes-large) to detect PERSON entities in the text/facts column
and adds a new 'has_person' column (1 if any PERSON entity found, 0 otherwise).

The script edits CSV files in place and creates backups.
"""

import argparse
import os
import shutil
from typing import List
import pandas as pd
from tqdm import tqdm

# Import Flair components
import flair
import torch
from flair.data import Sentence
from flair.models import SequenceTagger


class PersonNERTagger:
    """Simple PERSON entity tagger using Flair NER."""
    
    def __init__(self, model_name: str = "flair/ner-english-ontonotes-large", force_cpu: bool = False, use_gpu: bool = False):
        self.model_name = model_name
        self.force_cpu = force_cpu
        self.use_gpu = use_gpu
        self.tagger = None
        
    def _load_model(self):
        """Load the Flair NER model."""
        if self.tagger is not None:
            return
            
        print(f"Loading Flair NER model: {self.model_name}")
        
        if self.force_cpu:
            flair.device = torch.device('cpu')
            device = 'cpu'
        elif self.use_gpu and torch.cuda.is_available():
            device = 'cuda'
            flair.device = torch.device(device)
        else:
            device = 'cpu'
            flair.device = torch.device(device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.tagger = SequenceTagger.load(self.model_name).to(device)
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"NER model loaded on {device}")
    
    def has_person_entities(self, texts: List[str], batch_size: int = 32) -> List[bool]:
        """
        Check if texts contain PERSON entities.
        
        Args:
            texts: List of text strings to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of booleans indicating presence of PERSON entities
        """
        self._load_model()
        
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing NER batches"):
            batch_texts = texts[i:i + batch_size]
            batch_sentences = [Sentence(text) for text in batch_texts]
            
            # Run NER prediction
            self.tagger.predict(batch_sentences, verbose=False, mini_batch_size=batch_size)
            
            # Clear GPU cache after each batch to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Check for PERSON entities
            batch_results = []
            for sentence in batch_sentences:
                has_person = False
                for entity in sentence.get_spans('ner'):
                    # Check if any label is PERSON
                    for label in entity.get_labels():
                        if label.to_dict()['value'] == 'PERSON':
                            has_person = True
                            break
                    if has_person:
                        break
                batch_results.append(has_person)
            
            results.extend(batch_results)
        
        return results


def backup_file(file_path: str, backup_suffix: str = ".backup"):
    """Create a backup of the file."""
    backup_path = file_path + backup_suffix
    if os.path.exists(backup_path):
        print(f"Backup already exists: {backup_path}")
    else:
        shutil.copy2(file_path, backup_path)
        print(f"Created backup: {backup_path}")


def _safe_parse_facts_cell(cell: str):
    """Attempt to parse a stringified list-of-strings from the `facts` cell.

    Returns a list of fact strings. Falls back to splitting on newlines if parsing fails.
    """
    import ast

    if not isinstance(cell, str):
        return []

    cell = cell.strip()
    if cell == "":
        return []

    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass

    # Try JSON
    try:
        import json
        parsed = json.loads(cell)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass

    # Remove surrounding brackets if present
    if cell.startswith("[") and cell.endswith("]"):
        cell_inner = cell[1:-1].strip()
    else:
        cell_inner = cell

    # Split on patterns like "'1.  Text'" or numbered paragraphs '1.'
    import re
    parts = re.split(r"\n\s*\d{1,3}\.\s+|\n{2,}|\\n\\n", cell_inner)
    # If splitting produced one long element, try splitting on single newlines
    if len(parts) <= 1:
        parts = re.split(r"\n\s*\d{1,3}\.\s+|\n", cell_inner)

    # Clean quotes and whitespace
    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # strip leading/trailing quotes
        if (p.startswith("'") and p.endswith("'")) or (p.startswith('"') and p.endswith('"')):
            p = p[1:-1].strip()
        if p:
            cleaned.append(p)

    return cleaned


def process_csv_file(file_path: str, tagger: PersonNERTagger, text_column: str, batch_size: int, backup_suffix: str):
    """Process a single split CSV file: flatten (if needed) and annotate per-fact `has_person`.

    The function will:
    - Create a backup `file_path + backup_suffix`.
    - If the file already contains a per-fact column (named 'fact' or the provided `text_column`), annotate that file in place.
    - If the file contains a 'facts' column (stringified list), expand it into per-fact rows replicating other columns, then annotate the flattened DataFrame and overwrite the original `file_path` with the flattened version.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\nProcessing split file: {file_path}")

    # Load CSV
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")

    # Create backup of the original file before any modifications
    backup_file(file_path, backup_suffix)

    # Case 1: already flattened per-fact (has 'fact' column or the provided text_column)
    if 'fact' in df.columns or text_column in df.columns and 'facts' not in df.columns:
        # Determine the column to annotate
        if 'fact' in df.columns:
            fact_col = 'fact'
        else:
            fact_col = text_column

        print(f"Treating '{file_path}' as per-fact CSV (using column '{fact_col}'). Annotating in place.")
        texts = df[fact_col].fillna('').astype(str).tolist()
        person_flags = tagger.has_person_entities(texts, batch_size=batch_size)
        df['has_person'] = [1 if flag else 0 for flag in person_flags]
        df.to_csv(file_path, index=False)
        person_count = sum(person_flags)
        print(f"Results: {person_count}/{len(texts)} facts contain PERSON entities ({person_count/len(texts)*100:.1f}%)")
        print(f"Updated {file_path}")
        return

    # Case 2: input has a 'facts' column (stringified list) — flatten and annotate
    if 'facts' in df.columns:
        print(f"Found 'facts' column — flattening {file_path} into per-fact rows and annotating.")

        rows = []
        # We'll replicate other columns per fact where reasonable (labels etc.)
        other_cols = [c for c in df.columns if c != 'facts']

        for idx, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc='Flattening rows')):
            # Access the facts cell by column name via row (namedtuple) — fallback via index
            try:
                cell = getattr(row, 'facts')
            except Exception:
                cell = row[df.columns.get_loc('facts')]

            facts = _safe_parse_facts_cell(cell)
            if not facts:
                # create a placeholder empty fact row
                out = {c: getattr(row, c) if hasattr(row, c) else None for c in other_cols}
                out.update({'case_index': idx, 'fact_index': -1, 'fact': ''})
                rows.append(out)
            else:
                for fi, f in enumerate(facts):
                    out = {c: getattr(row, c) if hasattr(row, c) else None for c in other_cols}
                    out.update({'case_index': idx, 'fact_index': fi, 'fact': f})
                    rows.append(out)

        flat_df = pd.DataFrame(rows)

        # Annotate the flattened DataFrame
        fact_col = 'fact'
        texts = flat_df[fact_col].fillna('').astype(str).tolist()
        print(f"Analyzing {len(texts)} facts for PERSON entities...")
        person_flags = tagger.has_person_entities(texts, batch_size=batch_size)
        flat_df['has_person'] = [1 if flag else 0 for flag in person_flags]

        # Overwrite the original split file with the flattened, annotated CSV
        flat_df.to_csv(file_path, index=False)
        person_count = sum(person_flags)
        print(f"Flattened and updated {file_path}: {person_count}/{len(texts)} facts contain PERSON entities ({person_count/len(texts)*100:.1f}%)")
        return

    # Otherwise, no recognizable text/facts structure
    print(f"Error: {file_path} has no 'facts' column nor per-fact 'fact'/'{text_column}' column. Available columns: {list(df.columns)}")


def main():
    parser = argparse.ArgumentParser(description="Add PERSON entity detection to ECHR CSV files")
    parser.add_argument("--data_dir", type=str, default="./dataset", 
                       help="Directory containing echr_*.csv files")
    parser.add_argument("--backup_suffix", type=str, default=".backup",
                       help="Suffix for backup files")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for NER processing")
    parser.add_argument("--force_cpu", action="store_true", default=False,
                       help="Force NER model to use CPU (default: False)")
    parser.add_argument("--use_gpu", action="store_true", default=False,
                       help="Prefer GPU for NER if available (overrides --force_cpu)")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Name of text column to analyze")
    parser.add_argument("--files", nargs="+", default=["echr_train.csv", "echr_test.csv", "echr_validation.csv"],
                       help="CSV files to process")
    
    args = parser.parse_args()
    
    # Initialize NER tagger
    print("Initializing PERSON NER tagger...")
    tagger = PersonNERTagger(force_cpu=args.force_cpu, use_gpu=args.use_gpu)
    
    # Process each CSV file
    for filename in args.files:
        file_path = os.path.join(args.data_dir, filename)
        process_csv_file(file_path, tagger, args.text_column, args.batch_size, args.backup_suffix)
    
    print("\nAll files processed!")


if __name__ == "__main__":
    main()