"""
VLM-based auto-interpretation utilities for sparse features.

Provides functions to:
- Load vision-language models
- Format metadata for VLM consumption
- Generate natural language descriptions using simplified single-stage prompting
- Extract metadata statistics for results export
"""

# stdlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# third-party
import pandas as pd
import torch
from PIL import Image
from transformers import pipeline

# local
from src.clinical_region_mapping import map_organs_to_anatomical_regions


def load_vlm_model(
    model_name: str = "google/gemma-3-4b-it",
    dtype: torch.dtype = torch.bfloat16
) -> Any:
    """
    Load vision-language model pipeline.

    Uses device_map="auto" so weights are placed directly onto the target device
    during loading, avoiding the double-copy OOM that occurs when loading to CPU
    first and then calling .to("cuda").

    Args:
        model_name: HuggingFace model identifier (default: gemma-3-4b-it)
        dtype: Data type for model weights

    Returns:
        Transformers pipeline for image-text-to-text
    """
    print(f"Loading VLM: {model_name}")
    print(f"  Dtype: {dtype}")

    pipe = pipeline(
        "image-text-to-text",
        model=model_name,
        device_map="auto",
        dtype=dtype,
        image_processor_kwargs={"use_fast": False}  # Suppress warning about slow processor
    )

    print("[ok] VLM loaded successfully")
    return pipe


def clean_concept_description(text: str) -> str:
    """
    Remove common boilerplate phrases from VLM concept outputs.

    This is a safety net for cases where the VLM doesn't fully follow prompt instructions.

    Patterns removed:
    - "These images show/demonstrate/depict..."
    - "The images consistently/predominantly..."
    - "All images..."
    - Leading "Consistently", "Predominantly"

    Args:
        text: Raw VLM output

    Returns:
        Cleaned concept description
    """
    import re

    # Remove common boilerplate prefixes
    patterns = [
        r'^These (?:axial |coronal |sagittal )?(?:CT |MRI |images? )?(?:consistently |predominantly )?(?:show|demonstrate|depict|feature|present)\s+',
        r'^The (?:images? |scans? )?(?:consistently |predominantly )?(?:show|demonstrate|depict|feature|present)\s+',
        r'^All (?:images? |scans? )?(?:consistently |predominantly )?(?:show|demonstrate|depict|feature|present)\s+',
        r'^(?:Consistently|Predominantly),?\s+',
    ]

    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Capitalize first letter if needed
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned.strip()


def format_sample_metadata(sample: dict, sample_num: int) -> str:
    """
    Format metadata with fields ordered by importance for concept extraction.

    Priority order:
    1. Modality, Orientation (imaging characteristics)
    2. Sex, Age (patient demographics)
    3. Anatomical structures and organs (anatomical content)
    4. Pathology, Activation (clinical/feature metadata)

    Technical parameters (kVp, exposure, field strength, sequence) are excluded
    as they add noise rather than signal for semantic concept extraction.

    Args:
        sample: Sample dictionary with metadata
        sample_num: Sample number for display (1-indexed)

    Returns:
        Formatted metadata string
    """
    parts = [f"Sample {sample_num}:"]

    # Priority 1: Core imaging metadata
    if sample.get('modality'):
        parts.append(f"  Modality: {sample['modality'].upper()}")

    if sample.get('slice_orientation'):
        # Convert 'x', 'y', 'z' to human-readable
        orientation_map = {
            'x': 'Sagittal (x)',
            'y': 'Coronal (y)',
            'z': 'Axial (z)'
        }
        orientation = orientation_map.get(sample['slice_orientation'], sample['slice_orientation'])
        parts.append(f"  Orientation: {orientation}")

    # Priority 2: Patient demographics
    if sample.get('gender'):
        parts.append(f"  Sex: {sample['gender'].upper()}")

    if sample.get('age'):
        parts.append(f"  Age: {sample['age']} years")

    # Priority 3: Anatomical content
    if sample.get('organs'):
        organs = sample['organs']
        if isinstance(organs, list):
            organs_str = ', '.join(organs)
        else:
            organs_str = str(organs)
        parts.append(f"  Anatomical structures and organs ({len(organs)} total): {organs_str}")

        # Map raw organs to anatomical regions
        if isinstance(organs, list):
            anatomical_region = map_organs_to_anatomical_regions(organs)
            if anatomical_region:
                parts.append(f"  Anatomical regions: {', '.join(anatomical_region)}")

    # Priority 4: Clinical metadata
    if sample.get('pathology') and str(sample['pathology']).lower() not in ['none', 'nan', '']:
        parts.append(f"  Pathology: {sample['pathology']}")

    # Priority 5: Feature activation
    if sample.get('activation') is not None:
        parts.append(f"  Activation: {sample['activation']:.3f}")

    return '\n'.join(parts)


def format_all_samples_metadata(samples: list[dict]) -> str:
    """
    Format metadata for all samples.

    Args:
        samples: List of sample dictionaries

    Returns:
        Formatted metadata string for all samples
    """
    sample_blocks = []
    for i, sample in enumerate(samples, start=1):
        sample_blocks.append(format_sample_metadata(sample, i))

    return '\n\n'.join(sample_blocks)


def generate_feature_description(
    vlm_pipe: Any,
    images: list[Image.Image],
    samples: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 100,
    n_consistency_runs: int = 1
) -> str:
    """
    Generate natural language description using simplified single-stage prompting (v3.0).

    Philosophy:
    - Allow VLM to identify shared concepts naturally without forced narrowing
    - Trust VLM's perception of what's genuinely shared
    - Accept broader concepts if that's what the VLM perceives
    - Acknowledge that SAE monosemanticity may differ from human anatomical monosemanticity

    Approach:
    - Present all images simultaneously with their metadata
    - Ask: "What concept is shared across all these images?"
    - No constraints about structure counting, intersection vs union, etc.
    - Let the VLM describe what it actually sees

    Args:
        vlm_pipe: Loaded VLM pipeline
        images: List of PIL Images (top-N activating samples)
        samples: List of dicts with metadata for each image
        temperature: Sampling temperature (default: 0.3, slightly higher for natural language)
        max_tokens: Maximum tokens for output (default: 50, increased for flexible descriptions)
        n_consistency_runs: Number of runs for self-consistency voting (default: 1)
                           If > 1, runs multiple times and selects most frequent output

    Returns:
        Natural language description of the shared concept
    """
    # Format metadata for all samples
    metadata_text = format_all_samples_metadata(samples)

    # Build prompt
    prompt = f"""
Context: You are an expert radiologist analyzing a mono-semantic feature from a medical imaging sparse autoencoder (SAE) model.
A mono-semantic feature encodes a single concept that is shared across all samples that strongly activate it.

The situation: A mono-semantic feature has activated strongly on the {len(images)} samples below. Each sample includes metadata and a 2D CT or MRI slice.

Sample metadata:
{metadata_text}

Your task: Identify the single concept that is common to all samples. Provide a concise description focusing on anatomy or imaging characteristics.

Guidelines:
- Focus on the concept behind the mono-semantic feature. This is not a visual description of the images.
- Use the metadata (modality, orientation, sex, age, anatomical structures and organs, anatomical regions) to identify commonalities.
- Use the images as supporting evidence for the concept.
- Avoid listing all shared organs and anatomical structures. Group minor structures into their clinically relevant region (e.g. individual L-level vertebrae -> "lumbar spine", autochthon -> "paraspinal muscles")
- You may mention up to 3 key specific structures (no more!), e.g. "aorta", "liver", "gallbladder", "spinal canal", "skull", when they appear consistently and help distinguish this feature from other features of the same broad region.
- Include modality, slice orientation, and patient demographics when they are consistent across all samples.
- DO NOT mention lesions, masses, or pathology unless explicitly present and consistent across all samples in the metadata.
- Output a single phrase only.
- DO NOT use preambles like "These images show", "demonstrate", "consistently", "predominantly".

Output format: A single natural-language phrase describing the shared concept of the mono-semantic feature.
Examples: "CT of the spleen", "Coronal CT of the thorax and mediastinum", "Axial CT of the pelvis in a female patient", "Sagittal MRI of the lumbar spine and spinal canal in an elderly male patient".

Concept:"""

    # Prepare images and text for VLM
    # Build user content with images and text
    user_content = []
    for img in images:
        user_content.append({"type": "image", "url": img})
    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist analyzing monosemantic sparse features."}]},
        {"role": "user", "content": user_content}
    ]

    # Self-consistency voting: Run multiple times if requested
    outputs = []
    for run_idx in range(n_consistency_runs):
        output = vlm_pipe(
            text=messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )
        description = output[0]["generated_text"][-1]["content"].strip()
        # Clean up - remove quotes and boilerplate
        description = description.strip('"\'')
        description = clean_concept_description(description)
        outputs.append(description)

    # Select output based on consistency
    if n_consistency_runs == 1:
        # Single run - return directly
        final_description = outputs[0]
    else:
        # Multiple runs - select most frequent
        from collections import Counter
        output_counts = Counter(outputs)

        if len(output_counts) == 1:
            # All runs agree
            final_description = outputs[0]
        else:
            # Disagreement: select most frequent
            most_common = output_counts.most_common(2)
            if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                # Clear winner
                final_description = most_common[0][0]
            else:
                # Tie: select shortest (likely more concise)
                tied_outputs = [desc for desc, count in output_counts.items() if count == most_common[0][1]]
                final_description = min(tied_outputs, key=len)

    return final_description


def extract_metadata_statistics(samples: list[dict]) -> dict[str, Any]:
    """
    Extract summary statistics from sample metadata for results export.

    Args:
        samples: List of sample dicts

    Returns:
        Dict with aggregated metadata stats:
        - top_modalities: Most common modalities
        - top_orientations: Most common orientations
        - top_structures: Most common anatomical structures/organs
        - mean_activation: Mean activation value
    """
    from collections import Counter

    modalities = [s.get('modality', '') for s in samples]
    orientations = [s.get('slice_orientation', '') for s in samples]
    activations = [s.get('activation', 0.0) for s in samples]

    # Collect all organs/structures (flatten lists)
    all_organs = []
    for s in samples:
        organs = s.get('organs', [])
        if isinstance(organs, list):
            all_organs.extend(organs)
        elif organs:
            all_organs.append(organs)

    # Count frequencies
    modality_counts = Counter(modalities)
    orientation_counts = Counter(orientations)
    organ_counts = Counter(all_organs)

    return {
        'top_modalities': ', '.join([m for m, _ in modality_counts.most_common(3)]),
        'top_orientations': ', '.join([o for o, _ in orientation_counts.most_common(3)]),
        'top_structures': ', '.join([s for s, _ in organ_counts.most_common(5)]),
        'mean_activation': sum(activations) / len(activations) if activations else 0.0
    }


def rank_concepts_for_images(
    vlm_pipe: Any,
    images: list[Image.Image],
    concepts: list[str],
    samples: list[dict],
    temperature: float = 0.2
) -> dict[str, Any]:
    """
    Rank candidate concepts by how well they explain validation images (LLM-as-a-judge).

    This implements contrastive evaluation: the true concept must beat plausible alternatives.

    Args:
        vlm_pipe: Loaded VLM pipeline
        images: List of PIL Images (validation samples for target feature)
        concepts: List of 5 candidate concept descriptions (1 true + 4 distractors)
        samples: List of dicts with metadata for validation images
        temperature: Sampling temperature (default: 0.2 for deterministic ranking)

    Returns:
        Dict with:
        - ranking: List of concept indices in order (0-4, best to worst)
        - raw_output: Raw VLM output string
        - parse_success: Whether parsing succeeded
    """
    import random
    import re

    # Shuffle concepts to avoid position bias
    concept_labels = ['A', 'B', 'C', 'D', 'E']
    indices = list(range(len(concepts)))
    random.shuffle(indices)

    # Map shuffled positions to original indices
    shuffled_concepts = [concepts[i] for i in indices]
    label_to_original_idx = {concept_labels[i]: indices[i] for i in range(len(indices))}

    # Format metadata for context
    metadata_text = format_all_samples_metadata(samples)

    # Build ranking prompt
    concepts_text = '\n'.join([f"{concept_labels[i]}. {concept}" for i, concept in enumerate(shuffled_concepts)])

    prompt = f"""You are an expert radiologist evaluating sparse autoencoder features.

A monosemantic sparse feature has strongly activated on the {len(images)} validation samples below.

Sample metadata:
{metadata_text}

Below are 5 candidate concepts describing what this feature might detect:
{concepts_text}

Your task: Rank these concepts from BEST to WORST based on which best describes what this sparse feature has learned to detect.

Consider:
- Does the concept align with consistent patterns in the metadata (modality, orientation, sex, anatomical structures)?
- Does the concept capture what the feature detects, not just what the images show?
- Is the concept specific enough to distinguish this feature from random activations?

Output ONLY the ranking as letters separated by commas, from best to worst.
Example: "B, A, E, C, D"

Ranking:"""

    # Prepare VLM input
    user_content = []
    for img in images:
        user_content.append({"type": "image", "url": img})
    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist analyzing monosemantic sparse features."}]},
        {"role": "user", "content": user_content}
    ]

    # Generate ranking
    output = vlm_pipe(
        text=messages,
        max_new_tokens=50,
        temperature=temperature,
        do_sample=True if temperature > 0 else False
    )
    raw_output = output[0]["generated_text"][-1]["content"].strip().replace('\n', ' ')

    # Parse ranking
    # Expected format: "B, A, E, C, D" or "B A E C D" or similar
    # Extract letters A-E
    letters = re.findall(r'[A-E]', raw_output.upper())

    parse_success = False
    ranking = []

    if len(letters) == 5 and len(set(letters)) == 5:
        # Valid ranking - all 5 letters present and unique
        # Convert letters back to original indices
        ranking = [label_to_original_idx[letter] for letter in letters]
        parse_success = True
    else:
        # Parsing failed - return shuffled order as fallback
        ranking = indices
        parse_success = False

    return {
        'ranking': ranking,
        'raw_output': raw_output,
        'parse_success': parse_success,
        'shuffled_order': indices  # For debugging
    }


# ==================================================
# Language-Based Feature Matching
# ==================================================

def load_eligible_features(csv_path: Path, max_rank: int) -> pd.DataFrame:
    """
    Load feature descriptions filtered to true_concept_rank <= max_rank.

    Args:
        csv_path: Path to feature_descriptions.csv
        max_rank: Maximum true_concept_rank to include (1=best, 2=good, 3-5=failures)

    Returns:
        DataFrame of eligible features
    """
    df = pd.read_csv(csv_path)
    mask = df["true_concept_rank"] <= max_rank
    eligible = df.loc[mask].copy()
    print(
        f"Loaded {len(eligible)} eligible features "
        f"(true_concept_rank <= {max_rank}) from {len(df)} total"
    )
    return eligible


def match_query_to_features(
    vlm_pipe: Any,
    query_name: str,
    query_text: str,
    eligible: pd.DataFrame,
    n_features: int,
    temperature: float,
    vlm_model_name: str,
) -> dict:
    """
    Prompt VLM to select n_features best matching features for a natural language query.

    Args:
        vlm_pipe: Loaded VLM pipeline (from load_vlm_model)
        query_name: Short identifier for this query (used in error messages)
        query_text: Natural language query string
        eligible: DataFrame of candidate features with 'feature_idx' and 'description' columns
        n_features: Number of features to select
        temperature: Sampling temperature (0.1 for near-deterministic results)
        vlm_model_name: Model name stored in the output for auditing

    Returns:
        Dict with keys: name, query_text, matched_feature_indices, matched_descriptions,
        reasoning, vlm_model, timestamp - ready for JSON serialisation.

    Raises:
        ValueError: If VLM output cannot be parsed or returns wrong number of features,
                    or if a returned feature index is not in the eligible set.
    """
    # Use sequential numbering (1-based) to avoid VLM confusing list position
    # with the sparse feature_idx values (which can be large/sparse like 1063, 3293).
    seq_to_feature_idx: dict[int, int] = {}
    feature_lines = []
    for seq_num, (_, row) in enumerate(eligible.iterrows(), start=1):
        seq_to_feature_idx[seq_num] = int(row["feature_idx"])
        feature_lines.append(f"Feature [{seq_num}]: {row['description']}")
    features_block = "\n".join(feature_lines)

    prompt = (f"""
Context:
Below is a numbered list of monosemantic features from a sparse autoencoder trained on medical image embeddings.
Each feature description summarizes common visual and metadata patterns.

Feature list:
{features_block}

Task:
Select the n={n_features:d} features from the list that best match the clinical query:
{query_text}

Guidelines:
- Modality matching has the highest priority.
- Slice orientation matching has the second highest priority.
- Prefer exact anatomical matches over nearby regions.
- Prefer nearby region matches over demographic-only matches.
- Only match on dimensions the query explicitly specifies:
  - if the query does not mention orientation, any orientation is acceptable.
  - if the query does not mention modality, any modality is acceptable.
- If no acceptable match exists, select the closest available features.
- Do not select duplicates or near-identical concepts.
- Only use numbers from the feature list (1 to {len(feature_lines)}).

Output: Return EXACTLY n={n_features:d} feature numbers as a JSON array. No explanation.
{{"feature_indices": [{", ".join(f"NUM{i+1}" for i in range(n_features))}]}}
""")

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    valid_seq_nums = set(seq_to_feature_idx.keys())

    max_attempts = 3
    last_error: Exception | None = None
    indices: list[int] = []
    reasoning: str = ""

    for attempt in range(1, max_attempts + 1):
        output = vlm_pipe(
            text=messages,
            max_new_tokens=300,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        raw = output[0]["generated_text"][-1]["content"].strip()

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            last_error = ValueError(
                f"Could not parse JSON from VLM output for query '{query_name}': {raw!r}"
            )
            print(f"  [attempt {attempt}/{max_attempts}] Parse error, retrying...")
            continue

        result = json.loads(json_match.group())
        indices = [int(i) for i in result["feature_indices"]]
        reasoning = result.get("reasoning", "")

        if len(indices) != n_features:
            last_error = ValueError(
                f"Expected {n_features} features for '{query_name}', "
                f"got {len(indices)}: {indices}"
            )
            print(f"  [attempt {attempt}/{max_attempts}] Wrong count ({len(indices)}), retrying...")
            continue

        bad = [idx for idx in indices if idx not in valid_seq_nums]
        if bad:
            last_error = ValueError(
                f"Feature numbers {bad} returned by VLM not in valid range 1-{len(seq_to_feature_idx)} "
                f"(query: '{query_name}')"
            )
            print(f"  [attempt {attempt}/{max_attempts}] Invalid numbers {bad}, retrying...")
            continue

        last_error = None
        break

    if last_error is not None:
        raise last_error

    # Map sequential numbers back to actual feature_idx values
    indices = [seq_to_feature_idx[seq] for seq in indices]

    descriptions: list[str] = []
    for idx in indices:
        row = eligible[eligible["feature_idx"] == idx]
        descriptions.append(str(row.iloc[0]["description"]))

    return {
        "name": query_name,
        "query_text": query_text,
        "matched_feature_indices": indices,
        "matched_descriptions": descriptions,
        "reasoning": reasoning,
        "vlm_model": vlm_model_name,
        "timestamp": datetime.now().isoformat(),
    }
