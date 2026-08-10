from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT.parent / "Datasets" / "VitalDB"
CACHE_DIR = PROJECT_ROOT / "results" / "cache"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

TRAIN_MAT = DATA_ROOT / "VitalDB_Train_Subset.mat"
AAMI_TEST_MAT = DATA_ROOT / "VitalDB_AAMI_Test_Subset.mat"
AAMI_CAL_MAT = DATA_ROOT / "VitalDB_AAMI_Cal_Subset.mat"
CALBASED_TEST_MAT = DATA_ROOT / "VitalDB_CalBased_Test_Subset.mat"
CALFREE_TEST_MAT = DATA_ROOT / "VitalDB_CalFree_Test_Subset.mat"

DEFAULT_CHECKPOINT_SUBDIR = "full_v3"
DEFAULT_TRAIN_OUTPUT = CHECKPOINT_DIR / DEFAULT_CHECKPOINT_SUBDIR
DEFAULT_EVAL_OUTPUT = RESULTS_DIR / "eval"
