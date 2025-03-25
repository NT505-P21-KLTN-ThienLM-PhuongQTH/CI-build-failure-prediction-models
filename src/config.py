# utils/config.py
from typing import Dict, Any

CONFIG: Dict[str, Any] = {
    "NBR_REP": 6,
    "NBR_GEN": 5,
    "NBR_SOL": 5,
    "MAX_EVAL": 8,
    "WITH_SMOTE": True,
    "HYBRID_OPTION": True,
}

# Apply hybrid option logic
if CONFIG["HYBRID_OPTION"]:
    CONFIG["WITH_SMOTE"] = True