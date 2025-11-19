# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
from copy import deepcopy

import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm

from pii_leakage.arguments.attack_args import AttackArgs
from pii_leakage.arguments.config_args import ConfigArgs
from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.evaluation_args import EvaluationArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.attacks.attack_factory import AttackFactory
from pii_leakage.attacks.privacy_attack import PrivacyAttack, ExtractionAttack, ReconstructionAttack
from pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_leakage.models.language_model import LanguageModel
from pii_leakage.models.model_factory import ModelFactory
from pii_leakage.ner.pii_results import ListPII
from pii_leakage.ner.tagger_factory import TaggerFactory
from pii_leakage.utils.output import print_dict_highlighted
from pii_leakage.utils.set_ops import intersection


def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            DatasetArgs,
                                            AttackArgs,
                                            EvaluationArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def evaluate(model_args: ModelArgs,
             ner_args: NERArgs,
             dataset_args: DatasetArgs,
             attack_args: AttackArgs,
             eval_args: EvaluationArgs,
             env_args: EnvArgs,
             config_args: ConfigArgs):
    """ Evaluate a model and attack pair.
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        attack_args = config_args.get_attack_args()
        ner_args = config_args.get_ner_args()
        eval_args = config_args.get_evaluation_args()
        env_args = config_args.get_env_args()

    print_dict_highlighted(vars(attack_args))

    # Seed RNGs for reproducibility when requested
    seed = getattr(eval_args, 'seed', None)
    if seed is not None:
        import random as _random, numpy as _np
        _random.seed(seed)
        _np.random.seed(seed)
        try:
            import torch as _torch
            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
                _torch.backends.cudnn.deterministic = True
                _torch.backends.cudnn.benchmark = False
        except Exception:
            # torch may not be available in some environments; continue without torch seeding
            pass
        print(f"RNGs seeded with seed={seed}")

    # Load the target model (trained on private data)
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)

    # Create attack instance to check type before loading baseline
    attack: PrivacyAttack = AttackFactory.from_attack_args(attack_args, ner_args=ner_args, env_args=env_args)

    # Skip baseline model loading for reconstruction attacks since baseline filtering is disabled
    baseline_lm = None
    if not isinstance(attack, ReconstructionAttack):
        baseline_args = ModelArgs(**vars(model_args))
        baseline_args.model_ckpt = None
        
        # Force baseline to CPU to save GPU memory
        baseline_env_args = deepcopy(env_args)
        baseline_env_args.device = 'cpu'
        
        baseline_lm: LanguageModel = ModelFactory.from_model_args(baseline_args, env_args=baseline_env_args).load(verbose=True)

    train_dataset = DatasetFactory.from_dataset_args(dataset_args=dataset_args.set_split('train'), ner_args=ner_args)
    real_pii: ListPII = train_dataset.load_pii().flatten(attack_args.pii_class)
    print(f"Sample 20 real PII out of {len(real_pii.unique().mentions())}: {real_pii.unique().mentions()[:20]}")

    if isinstance(attack, ExtractionAttack):
        # Compute Precision/Recall for the extraction attack.
        print("Generating text with target model...")
        generated_pii = set(attack.attack(lm).keys())
        
        if baseline_lm is not None:
            print("Generating text with baseline model...")
            baseline_pii = set(attack.attack(baseline_lm).keys())
        else:
            print("Baseline model not loaded, skipping baseline evaluation...")
            baseline_pii = set()
        
        # Unload models to free GPU memory before NER analysis
        print("Unloading models to free memory for NER...")
        if hasattr(lm, 'unload'):
            lm.unload()
        if baseline_lm is not None and hasattr(baseline_lm, 'unload'):
            baseline_lm.unload()
            
        # Clear GPU cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_memory_gb = free_memory / (1024**3)
            print(f"Memory freed. Available GPU memory: {free_memory_gb:.2f} GB")
        
        real_pii_set = set(real_pii.unique().mentions())

        # TEMPORARILY DISABLED: Remove baseline leakage (comment out to re-enable)
        # leaked_pii = generated_pii.difference(baseline_pii)
        
        # Use generated PII directly (without baseline filtering)
        leaked_pii = generated_pii

        print(f"Generated: {len(generated_pii)}")
        print(f"Baseline:  {len(baseline_pii)}")
        print(f"Leaked:    {len(leaked_pii)} (without baseline filtering)")

        print(f"Precision: {100 * len(real_pii_set.intersection(leaked_pii)) / len(leaked_pii):.2f}%")
        print(f"Recall:    {100 * len(real_pii_set.intersection(leaked_pii)) / len(real_pii):.2f}%")

    elif isinstance(attack, ReconstructionAttack):
        # Compute accuracy for the reconstruction/inference attack.
        idx = random.sample(range(len(train_dataset)), len(train_dataset))
        dataset = train_dataset.select(idx)  # dict with 'text': seq and 'entity_class': 'ListPII (as a str)'

        tagger = TaggerFactory.from_ner_args(ner_args, env_args=env_args)
        
        results_data = {"Predicted": [], "True": [], "Sequence": []}
        
        with tqdm(total=eval_args.num_sequences, desc="Evaluate Reconstruction") as pbar:
            y_preds, y_trues = [], []
            sequences_processed = 0
            sequences_examined = 0
            
            for seq in dataset:
                sequences_examined += 1
                    
                if sequences_processed >= eval_args.num_sequences:
                    break

                # 1. Assert that the sequence has at least one PII
                try:
                    pii = tagger.analyze(seq['text']).get_by_entity_class(attack_args.pii_class).unique()
                    pii = ListPII(data=[p for p in pii if len(p.text) > 1])  # min chars for PII (reduced from 3 to allow short names)
                    if len(pii) == 0:
                        if sequences_examined % 100 == 0:  # Progress: print every 100 sequences
                            print(f"\n⚠️  Progress: examined {sequences_examined} sequences, found {sequences_processed} valid (no {attack_args.pii_class} entities)")
                        continue
                    
                    if sequences_examined % 50 == 0:  # Progress: print when we find entities
                        print(f"\n✅ Progress: sequence {sequences_examined} has {len(pii)} {attack_args.pii_class} entities")
                        
                except Exception as e:
                    print(f"\n⚠️  Error analyzing sequence: {e}")
                    continue

                # 2. Randomly sample one target PII
                target_pii = random.sample(pii.mentions(), 1)[0]

                # 3. Replace the target PII with <T-MASK> and other PII with <MASK>
                target_sequence = seq['text'].replace(target_pii, '<T-MASK>', 1)
                for pii_mention in pii.mentions():
                    target_sequence = target_sequence.replace(pii_mention, '<MASK>')

                # 4. Randomly sample candidates
                assert eval_args.num_candidates <= len(real_pii.unique().mentions()), f"Not enough real candidates " \
                                                                                        f"({len(real_pii.unique().mentions())}) to accomodate candidate size ({eval_args.num_candidates})."
                candidate_pii = random.sample(real_pii.unique().mentions(), eval_args.num_candidates - 1) + [
                    target_pii]
                random.shuffle(candidate_pii)  # shuffle to ensure there is no positional leakage

                # 5. Run the reconstruction attack
                try:
                    # Set attack_args.target_sequence before calling attack
                    attack.attack_args.target_sequence = target_sequence
                    result = attack.attack(lm, pii_candidates=candidate_pii, verbose=False)
                    predicted_target_pii = result[min(result.keys())]
                    
                    # Memory cleanup after each attack
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"\n⚠️  Error in target model attack: {e}")
                    # Memory cleanup on error too
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                # BASELINE EVALUATION DISABLED FOR PERFORMANCE
                # Since baseline filtering is disabled, skip baseline evaluation entirely
                # 6. Evaluate baseline leakage with timeout protection
                # try:
                #     baseline_result = attack.attack(baseline_lm, target_sequence=target_sequence, pii_candidates=candidate_pii, verbose=False)
                #     baseline_target_pii = baseline_result[min(baseline_result.keys())]
                # except Exception as e:
                #     print(f"\n⚠️  Error in baseline model attack: {e}")
                #     continue

                # TEMPORARILY DISABLED: Baseline leakage check (remove this to re-enable)
                # if baseline_target_pii == predicted_target_pii:
                #     # Baseline leakage because public model has the same prediction. Skip
                #     if sequences_examined % 100 == 0:  # Progress: print every 100 sequences
                #         print(f"\n⚠️  Progress: examined {sequences_examined} sequences, found {sequences_processed} valid (skipping baseline leakage)")
                #     continue

                y_preds += [predicted_target_pii]
                y_trues += [target_pii]
                
                # Store in dataframe
                results_data["Predicted"].append(predicted_target_pii)
                results_data["True"].append(target_pii)
                results_data["Sequence"].append(target_sequence)
                
                sequences_processed += 1

                acc = np.mean([1 if y_preds[i] == y_trues[i] else 0 for i in range(len(y_preds))])
                pbar.set_description(f"Evaluate Reconstruction: Accuracy: {100 * acc:.2f}% ({sequences_processed}/{eval_args.num_sequences})")
                pbar.update(1)
                
            print(f"\nCompleted: {sequences_processed} valid sequences processed from {sequences_examined} examined")
        
        # Save results to CSV
        if sequences_processed > 0:
            results_df = pd.DataFrame(results_data)
            output_path = "/home/dpereira/CB-LLMs/disentangling/work/check_vib.csv"
            results_df.to_csv(output_path, index=False)
            print(f"\nSaved {len(results_df)} predictions to {output_path}")
    else:
        raise ValueError(f"Unknown attack type: {type(attack)}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(*parse_args())
# ----------------------------------------------------------------------------
