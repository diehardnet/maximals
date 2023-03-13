RANDOM_INT_LIMIT = 65535

MAXIMUM_ERRORS_PER_ITERATION = 512
MAXIMUM_INFOS_PER_ITERATION = 512

# Device capability for pytorch
MINIMUM_DEVICE_CAPABILITY = 5  # Maxwell

CLASSIFICATION_CRITICAL_TOP_K = 1
# FORCE the gpu to be present
DEVICE = "cuda:0"

RESNET50_IMAGENET_1K_V2_BASE = "resnet50_imagenet1k_v2_base"
CNN_CONFIGS = [
    # Baseline
    RESNET50_IMAGENET_1K_V2_BASE,

]

# Classification ViTs
VITS_BASE_PATCH16_224 = "vit_base_patch16_224"
VITS_BASE_PATCH16_384 = "vit_base_patch32_384"
VITS_BASE_PATCH32_224_SAM = "vit_base_patch32_224_sam"
VITS_BASE_RESNET50_384 = "vit_base_resnet50_384"
VITS_CLASSIFICATION_CONFIGS = [
    VITS_BASE_PATCH16_224,
    VITS_BASE_PATCH16_384,
    VITS_BASE_PATCH32_224_SAM,
    VITS_BASE_RESNET50_384
]

# Set the supported goals
CLASSIFICATION = "classify"
SEGMENTATION = "segmentation"

DNN_GOAL = {
    # Classification nets
    **{k: CLASSIFICATION for k in VITS_CLASSIFICATION_CONFIGS},
    # Segmentation nets
}

# Error threshold for the test
DNN_THRESHOLD = {
    CLASSIFICATION: 0.01,
    SEGMENTATION: 0.01
}

ITERATION_INTERVAL_LOG_HELPER_PRINT = {
    # imagenet not so small
    **{k: 10 for k in CNN_CONFIGS},
    **{k: 1 for k in VITS_CLASSIFICATION_CONFIGS},
    # Segmentation nets, huge
}

IMAGENET = "imagenet"
COCO = "coco"

CLASSES = {
    IMAGENET: 1000
}

IMAGENET_DATASET_DIR = "/home/carol/ILSVRC2012"
COCO_DATASET_DIR = "/home/carol/coco"
COCO_DATASET_VAL = f"{COCO_DATASET_DIR}/val2017"
COCO_DATASET_ANNOTATIONS = f"{COCO_DATASET_DIR}/annotations/instances_val2017.json"

# File to save last status of the benchmark when log helper not active
TMP_CRASH_FILE = "/tmp/maximal_crash_file.txt"
