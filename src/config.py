import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_ranker.pkl")

BEST_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'eval_at': 5,
    'boosting_type': 'gbdt',
    'random_state': 42,
    'learning_rate': 0.168,
    'num_leaves': 47,
    'lambda_l1': 1.818,
    'lambda_l2': 1.824,
    'bagging_fraction': 0.652,
    'feature_fraction': 0.762,
    'verbosity': -1
}