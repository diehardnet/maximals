RANDOM_INT_LIMIT = 65535

MAXIMUM_ERRORS_PER_ITERATION = 512
MAXIMUM_INFOS_PER_ITERATION = 512

# Device capability for pytorch
MINIMUM_DEVICE_CAPABILITY = 5  # Maxwell

CLASSIFICATION_CRITICAL_TOP_K = 1
# FORCE the gpu to be present
DEVICE = "cuda:0"

# Classification CNNs
RESNET50D_IMAGENET_TIMM = "resnet50d"
RESNET200D_IMAGENET_TIMM = "resnet200d"
EFFICIENTNET_B7_TIMM = "tf_efficientnet_b7"
CNN_CONFIGS = [
    RESNET50D_IMAGENET_TIMM,
    RESNET200D_IMAGENET_TIMM,
    EFFICIENTNET_B7_TIMM
]

# Classification ViTs
# Base from the paper
VIT_BASE_PATCH16_224 = "vit_base_patch16_224"
VIT_BASE_PATCH16_384 = "vit_base_patch32_384"
# Large models
VIT_LARGE_PATCH14_CLIP_336 = "vit_large_patch14_clip_336"
VIT_LARGE_PATCH14_CLIP_224 = "vit_large_patch14_clip_224"
# Huge models
VIT_HUGE_PATCH14_CLIP_336 = "vit_huge_patch14_clip_336"
VIT_HUGE_PATCH14_CLIP_224 = "vit_huge_patch14_clip_224"
# Max vit
MAXVIT_XLARGE_TF_384 = 'maxvit_xlarge_tf_384'
MAXVIT_XLARGE_TF_512 = 'maxvit_xlarge_tf_512'
# Davit
DAVIT_GIANT = 'davit_giant'
DAVIT_HUGE = 'davit_huge'
# SwinV2
SWINV2_BASE_WINDOW12TO16_192to256_22KFT1K = 'swinv2_base_window12to16_192to256_22kft1k'
SWINV2_BASE_WINDOW12TO24_192to384_22KFT1K = 'swinv2_base_window12to24_192to384_22kft1k'

SWINV2_LARGE_WINDOW12TO16_192to256_22KFT1K = 'swinv2_large_window12to16_192to256_22kft1k'
SWINV2_LARGE_WINDOW12TO24_192to384_22KFT1K = 'swinv2_large_window12to24_192to384_22kft1k'

VIT_CLASSIFICATION_CONFIGS = [
    VIT_BASE_PATCH16_224, VIT_BASE_PATCH16_384,
    VIT_LARGE_PATCH14_CLIP_336, VIT_LARGE_PATCH14_CLIP_224,
    VIT_HUGE_PATCH14_CLIP_336, VIT_HUGE_PATCH14_CLIP_224,
    # MAXVIT_XLARGE_TF_384, MAXVIT_XLARGE_TF_512,
    # DAVIT_GIANT, DAVIT_HUGE, -- Not working
    SWINV2_LARGE_WINDOW12TO16_192to256_22KFT1K, SWINV2_LARGE_WINDOW12TO24_192to384_22KFT1K,
    SWINV2_BASE_WINDOW12TO16_192to256_22KFT1K, SWINV2_BASE_WINDOW12TO24_192to384_22KFT1K,
]

ALL_POSSIBLE_MODELS = CNN_CONFIGS + VIT_CLASSIFICATION_CONFIGS

# Set the supported goals
CLASSIFICATION = "classify"
SEGMENTATION = "segmentation"

DNN_GOAL = {
    # Classification CNNs
    **{k: CLASSIFICATION for k in CNN_CONFIGS},
    # Classification transformer
    **{k: CLASSIFICATION for k in VIT_CLASSIFICATION_CONFIGS},
    # Segmentation nets
}

# Error threshold for the test
DNN_THRESHOLD = {
    CLASSIFICATION: 0.01,
    SEGMENTATION: 0.01
}

ITERATION_INTERVAL_LOG_HELPER_PRINT = {
    # imagenet not so small
    **{k: 1 for k in CNN_CONFIGS},
    **{k: 1 for k in VIT_CLASSIFICATION_CONFIGS},
    # Segmentation nets, huge
}

IMAGENET = "imagenet"
COCO = "coco"
DATASETS = {
    CLASSIFICATION: IMAGENET,
    SEGMENTATION: COCO
}

CLASSES = {
    IMAGENET: 1000
}

IMAGENET_DATASET_DIR = "/home/carol/ILSVRC2012"
COCO_DATASET_DIR = "/home/carol/coco"
COCO_DATASET_VAL = f"{COCO_DATASET_DIR}/val2017"
COCO_DATASET_ANNOTATIONS = f"{COCO_DATASET_DIR}/annotations/instances_val2017.json"

# File to save last status of the benchmark when log helper not active
TMP_CRASH_FILE = "/tmp/maximal_crash_file.txt"

# TensorRT file pattern
TENSORRT_FILE_POSFIX = "_tensorrt.ts"
