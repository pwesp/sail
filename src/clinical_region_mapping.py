"""
Deterministic mapping from TotalSegmentator organ labels to anatomical regions.

Provides the VLM with high-level anatomical vocabulary (e.g., "lumbar spine")
alongside the raw organ labels (e.g., "vertebrae_L1", "vertebrae_L3").
"""

# Every raw organ label maps to exactly one anatomical region.
ORGAN_TO_ANATOMICAL_REGION: dict[str, str] = {
    # Cervical spine
    'vertebrae_C1': 'cervical spine',
    'vertebrae_C2': 'cervical spine',
    'vertebrae_C3': 'cervical spine',
    'vertebrae_C4': 'cervical spine',
    'vertebrae_C5': 'cervical spine',
    'vertebrae_C6': 'cervical spine',
    'vertebrae_C7': 'cervical spine',

    # Thoracic spine
    'vertebrae_T1': 'thoracic spine',
    'vertebrae_T2': 'thoracic spine',
    'vertebrae_T3': 'thoracic spine',
    'vertebrae_T4': 'thoracic spine',
    'vertebrae_T5': 'thoracic spine',
    'vertebrae_T6': 'thoracic spine',
    'vertebrae_T7': 'thoracic spine',
    'vertebrae_T8': 'thoracic spine',
    'vertebrae_T9': 'thoracic spine',
    'vertebrae_T10': 'thoracic spine',
    'vertebrae_T11': 'thoracic spine',
    'vertebrae_T12': 'thoracic spine',

    # Lumbar spine
    'vertebrae_L1': 'lumbar spine',
    'vertebrae_L2': 'lumbar spine',
    'vertebrae_L3': 'lumbar spine',
    'vertebrae_L4': 'lumbar spine',
    'vertebrae_L5': 'lumbar spine',

    # Sacrum
    'sacrum': 'sacrum',
    'vertebrae_S1': 'sacrum',

    # Spinal canal
    'spinal_cord': 'spinal canal',
    'intervertebral_discs': 'spinal canal',
    'vertebrae': 'spinal canal',

    # Paraspinal muscles
    'autochthon_left': 'paraspinal muscles',
    'autochthon_right': 'paraspinal muscles',

    # Brain
    'brain': 'brain',

    # Skull
    'skull': 'skull',

    # Thyroid
    'thyroid_gland': 'thyroid',

    # Lungs
    'lung_left': 'lungs',
    'lung_right': 'lungs',
    'lung_lower_lobe_left': 'lungs',
    'lung_lower_lobe_right': 'lungs',
    'lung_middle_lobe_right': 'lungs',
    'lung_upper_lobe_left': 'lungs',
    'lung_upper_lobe_right': 'lungs',

    # Heart
    'heart': 'heart',
    'atrial_appendage_left': 'heart',
    'pulmonary_vein': 'heart',

    # Mediastinum
    'trachea': 'mediastinum',
    'esophagus': 'mediastinum',
    'brachiocephalic_trunk': 'mediastinum',
    'brachiocephalic_vein_left': 'mediastinum',
    'brachiocephalic_vein_right': 'mediastinum',
    'superior_vena_cava': 'mediastinum',
    'subclavian_artery_left': 'mediastinum',
    'subclavian_artery_right': 'mediastinum',
    'common_carotid_artery_left': 'mediastinum',
    'common_carotid_artery_right': 'mediastinum',

    # Rib cage
    'sternum': 'rib cage',
    'costal_cartilages': 'rib cage',
    'rib_left_1': 'rib cage',
    'rib_left_2': 'rib cage',
    'rib_left_3': 'rib cage',
    'rib_left_4': 'rib cage',
    'rib_left_5': 'rib cage',
    'rib_left_6': 'rib cage',
    'rib_left_7': 'rib cage',
    'rib_left_8': 'rib cage',
    'rib_left_9': 'rib cage',
    'rib_left_10': 'rib cage',
    'rib_left_11': 'rib cage',
    'rib_left_12': 'rib cage',
    'rib_right_1': 'rib cage',
    'rib_right_2': 'rib cage',
    'rib_right_3': 'rib cage',
    'rib_right_4': 'rib cage',
    'rib_right_5': 'rib cage',
    'rib_right_6': 'rib cage',
    'rib_right_7': 'rib cage',
    'rib_right_8': 'rib cage',
    'rib_right_9': 'rib cage',
    'rib_right_10': 'rib cage',
    'rib_right_11': 'rib cage',
    'rib_right_12': 'rib cage',

    # Shoulder girdle
    'scapula_left': 'shoulder girdle',
    'scapula_right': 'shoulder girdle',
    'humerus_left': 'shoulder girdle',
    'humerus_right': 'shoulder girdle',
    'clavicula_left': 'shoulder girdle',
    'clavicula_right': 'shoulder girdle',

    # Liver
    'liver': 'liver',

    # Gallbladder
    'gallbladder': 'gallbladder',

    # Spleen
    'spleen': 'spleen',

    # Pancreas
    'pancreas': 'pancreas',

    # Stomach
    'stomach': 'stomach',

    # Kidneys
    'kidney_left': 'kidneys',
    'kidney_right': 'kidneys',
    'kidney_cyst_left': 'kidneys',
    'kidney_cyst_right': 'kidneys',

    # Adrenal glands
    'adrenal_gland_left': 'adrenal glands',
    'adrenal_gland_right': 'adrenal glands',

    # Bowel
    'colon': 'bowel',
    'small_bowel': 'bowel',
    'duodenum': 'bowel',

    # Abdominal vasculature
    'aorta': 'abdominal vasculature',
    'inferior_vena_cava': 'abdominal vasculature',
    'portal_vein_and_splenic_vein': 'abdominal vasculature',

    # Pelvic vasculature
    'iliac_artery_left': 'pelvic vasculature',
    'iliac_artery_right': 'pelvic vasculature',
    'iliac_vena_left': 'pelvic vasculature',
    'iliac_vena_right': 'pelvic vasculature',

    # Bony pelvis
    'hip_left': 'bony pelvis',
    'hip_right': 'bony pelvis',

    # Proximal femur
    'femur_left': 'proximal femur',
    'femur_right': 'proximal femur',

    # Gluteal muscles
    'gluteus_maximus_left': 'gluteal muscles',
    'gluteus_maximus_right': 'gluteal muscles',
    'gluteus_medius_left': 'gluteal muscles',
    'gluteus_medius_right': 'gluteal muscles',
    'gluteus_minimus_left': 'gluteal muscles',
    'gluteus_minimus_right': 'gluteal muscles',

    # Hip flexors
    'iliopsoas_left': 'hip flexors',
    'iliopsoas_right': 'hip flexors',

    # Urinary bladder
    'urinary_bladder': 'urinary bladder',

    # Prostate
    'prostate': 'prostate',
}


# Canonical organ list from src/dataloading.py - assert completeness at import time.
_CANONICAL_ORGAN_COLUMNS = {
    'adrenal_gland_left', 'adrenal_gland_right', 'aorta', 'atrial_appendage_left', 'autochthon_left', 'autochthon_right',
    'brachiocephalic_trunk', 'brachiocephalic_vein_left', 'brachiocephalic_vein_right', 'brain', 'clavicula_left', 'clavicula_right',
    'colon', 'common_carotid_artery_left', 'common_carotid_artery_right', 'costal_cartilages', 'duodenum', 'esophagus',
    'femur_left', 'femur_right', 'gallbladder', 'gluteus_maximus_left', 'gluteus_maximus_right', 'gluteus_medius_left',
    'gluteus_medius_right', 'gluteus_minimus_left', 'gluteus_minimus_right', 'heart', 'hip_left', 'hip_right',
    'humerus_left', 'humerus_right', 'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left', 'iliac_vena_right',
    'iliopsoas_left', 'iliopsoas_right', 'inferior_vena_cava', 'intervertebral_discs', 'kidney_cyst_left', 'kidney_cyst_right',
    'kidney_left', 'kidney_right', 'liver', 'lung_left', 'lung_lower_lobe_left', 'lung_lower_lobe_right',
    'lung_middle_lobe_right', 'lung_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right', 'pancreas', 'portal_vein_and_splenic_vein',
    'prostate', 'pulmonary_vein', 'rib_left_1', 'rib_left_10', 'rib_left_11', 'rib_left_12',
    'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7',
    'rib_left_8', 'rib_left_9', 'rib_right_1', 'rib_right_10', 'rib_right_11', 'rib_right_12',
    'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5', 'rib_right_6', 'rib_right_7',
    'rib_right_8', 'rib_right_9', 'sacrum', 'scapula_left', 'scapula_right', 'skull',
    'small_bowel', 'spinal_cord', 'spleen', 'sternum', 'stomach', 'subclavian_artery_left',
    'subclavian_artery_right', 'superior_vena_cava', 'thyroid_gland', 'trachea', 'urinary_bladder', 'vertebrae',
    'vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4', 'vertebrae_C5', 'vertebrae_C6',
    'vertebrae_C7', 'vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5',
    'vertebrae_S1', 'vertebrae_T1', 'vertebrae_T10', 'vertebrae_T11', 'vertebrae_T12', 'vertebrae_T2',
    'vertebrae_T3', 'vertebrae_T4', 'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7', 'vertebrae_T8',
    'vertebrae_T9',
}

_mapped_organs = set(ORGAN_TO_ANATOMICAL_REGION.keys())
_missing = _CANONICAL_ORGAN_COLUMNS - _mapped_organs
_extra = _mapped_organs - _CANONICAL_ORGAN_COLUMNS
assert not _missing, f"Organs missing from ORGAN_TO_CLINICAL_REGION: {sorted(_missing)}"
assert not _extra, f"Extra organs in ORGAN_TO_CLINICAL_REGION not in canonical list: {sorted(_extra)}"


def map_organs_to_anatomical_regions(organs: list[str]) -> list[str]:
    """
    Map raw organ labels to deduplicated, sorted anatomical regions.

    Args:
        organs: List of raw TotalSegmentator organ labels
                (e.g., ["vertebrae_L1", "autochthon_left"])

    Returns:
        Sorted list of unique anatomical region names
        (e.g., ["lumbar spine", "paraspinal muscles"])
    """
    regions = {ORGAN_TO_ANATOMICAL_REGION[organ] for organ in organs if organ in ORGAN_TO_ANATOMICAL_REGION}
    return sorted(regions)
