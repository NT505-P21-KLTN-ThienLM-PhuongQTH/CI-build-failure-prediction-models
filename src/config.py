CONFIG = {
    'NBR_REP': 6,
    'NBR_GEN': 5,
    'NBR_SOL': 5,
    'MAX_EVAL': 8,
    'WITH_SMOTE': True,
    'HYBRID_OPTION': True
}

if CONFIG['HYBRID_OPTION']:
    CONFIG['WITH_SMOTE'] = True