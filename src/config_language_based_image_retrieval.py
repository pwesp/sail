"""
Query configuration for language-based image retrieval.

Edit QUERIES to add, remove, or modify queries.
Each query has:
  name  - snake_case identifier used for subdirectory names and result tracking
  text  - natural language description of the images to retrieve

Feature matching (query -> feature indices) is performed separately by
match_queries_to_features.py and saved to query_feature_matching.json.
"""

QUERIES = [
    # -- Anatomy + orientation --------------------------------------------------
    {
        "name": "ct_thorax_coronal",
        "text": "Coronal CT of the thorax and rib cage",
    },
    {
        "name": "ct_spine_sagittal",
        "text": "Sagittal CT of the thoracolumbar spine and spinal canal",
    },
    {
        "name": "ct_lumbar_axial",
        "text": "Axial CT of the lumbar spine and spinal canal",
    },
    {
        "name": "ct_pelvis_coronal",
        "text": "Coronal CT of the bony pelvis and lumbar spine",
    },
    {
        "name": "ct_abdomen_pelvis_coronal",
        "text": "Coronal CT of the abdomen and pelvis",
    },
    {
        "name": "ct_skull_axial",
        "text": "Axial CT of the skull and brain",
    },
    # -- Anatomy + orientation + demographics ----------------------------------
    {
        "name": "ct_female_pelvis",
        "text": "Axial CT of the female pelvis",
    },
    {
        "name": "ct_male_pelvis",
        "text": "Axial CT of the male pelvis",
    },
    {
        "name": "ct_thorax_female",
        "text": "Axial CT of the thorax and mediastinum in a female patient",
    },
    {
        "name": "ct_thorax_male",
        "text": "Axial CT of the thorax and mediastinum in a male patient",
    },
    {
        "name": "ct_abdomen_elderly",
        "text": "Axial CT of the abdomen and retroperitoneum in an elderly patient",
    },
    # -- MRI -------------------------------------------------------------------
    {
        "name": "mri_pelvis",
        "text": "Axial MRI of the pelvis",
    },
]
