"""
Configuration for downstream classification tasks.

This module contains all configuration constants for elastic-net logistic regression
on dense embeddings and sparse SAE features.
"""

# ==================================================
# Base Model Paths
# ==================================================

BASE_MODEL_PATHS = {
    'biomedparse': {
        'dense_train': 'data/total_seg_biomedparse_encodings/total_seg_biomedparse_encodings_train.parquet',
        'dense_val': 'data/total_seg_biomedparse_encodings/total_seg_biomedparse_encodings_valid.parquet',
        'dense_test': 'data/total_seg_biomedparse_encodings/total_seg_biomedparse_encodings_test.parquet',
        'sparse_base': 'results/total_seg_sparse_features/biomedparse_matryoshka_sae',
    },
    'biomedparse_random': {
        'dense_train': 'data/total_seg_biomedparse_random_encodings/total_seg_biomedparse_random_encodings_train.parquet',
        'dense_val': 'data/total_seg_biomedparse_random_encodings/total_seg_biomedparse_random_encodings_valid.parquet',
        'dense_test': 'data/total_seg_biomedparse_random_encodings/total_seg_biomedparse_random_encodings_test.parquet',
        'sparse_base': 'results/total_seg_sparse_features/biomedparse_random_matryoshka_sae',
    },
    'dinov3': {
        'dense_train': 'data/total_seg_dinov3_encodings/total_seg_dinov3_encodings_train.parquet',
        'dense_val': 'data/total_seg_dinov3_encodings/total_seg_dinov3_encodings_valid.parquet',
        'dense_test': 'data/total_seg_dinov3_encodings/total_seg_dinov3_encodings_test.parquet',
        'sparse_base': 'results/total_seg_sparse_features/dinov3_matryoshka_sae',
    }
}

RESULTS_BASE_DIR = "results/linear_probes"

# ==================================================
# Target Groups
# ==================================================

TARGET_GROUPS = {
    # Multi-class groups (targets are mutually exclusive)
    'modality': {
        'type': 'multiclass',
        'targets': ['mri', 'ct'],
        'description': 'Imaging modality classification'
    },
    'sex': {
        'type': 'multiclass',
        'targets': ['f', 'm'],
        'description': 'Patient sex classification'
    },
    'slice_orientation': {
        'type': 'multiclass',
        'targets': ['x', 'y', 'z'],
        'description': 'Slice orientation (axial/sagittal/coronal)'
    },
    'age_group': {
        'type': 'multiclass',
        'targets': ['<40', '>=80'],
        'description': 'Patient age group (young vs elderly)'
    },
    'kvp': {
        'type': 'multiclass',
        'targets': ['80.0', '120.0'],
        'description': 'X-ray tube voltage'
    },
    'institute': {
        'type': 'multiclass',
        'targets': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'description': 'Institution'
    },

    # Multi-label groups (targets can co-occur, use binary relevance)
    'manufacturer': {
        'type': 'multilabel',
        'targets': ['ge', 'philips', 'siemens'],
        'description': 'Scanner manufacturer'
    },
    'scanner_model': {
        'type': 'multilabel',
        'targets': ['Achieva', 'Avanto_fit', 'Skyra'],
        'description': 'Scanner model'
    },
    'organs_abdominal': {
        'type': 'multilabel',
        'targets': ['gallbladder', 'kidney_left', 'liver', 'spleen', 'prostate'],
        'description': 'Abdominal organs'
    },
    'organs_head_and_neck': {
        'type': 'multilabel',
        'targets': ['brain'],
        'description': 'Head and neck organs'
    },
    'organs_thoracic': {
        'type': 'multilabel',
        'targets': ['heart', 'sternum', 'clavicula_left', 'clavicula_right'],
        'description': 'Thoracic structures'
    },
    'pathology': {
        'type': 'multilabel',
        'targets': ['inflammation', 'tumor', 'trauma'],
        'description': 'Pathological findings'
    },
}

# ==================================================
# Hyperparameter Configuration
# ==================================================

# Feature importance
TOP_K_FEATURES = 100  # Number of top features to save (coefficient-based importance)

# Visualization
TOP_K_FEATURES_VIZ = 10  # Number of top features to visualize in plots
N_SAMPLES_VIZ = 5        # Number of top activating samples per feature

# Training data limits
MAX_TRAINING_SAMPLES = 10000   # Maximum training samples after subsampling

# Model training
LOGISTIC_MAX_ITER = 1000

# Regression parameters
LOGISTIC_C = 1.0          # Regularization strength
LOGISTIC_L1_RATIO = 0.5   # 50% Lasso, 50% Ridge (elastic-net)
