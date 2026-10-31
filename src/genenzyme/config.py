import os
# -----------------------
# TOKENS
# -----------------------
SPECIAL_TOKENS = {
    "<START>": 0,
    "<END>": 1,
    "<PAD>": 2,
    "<UNK>": 3,
    "<PREDICT_REAC>": 4,
    "<PREDICT_PROD>": 5,
}

# -----------------------
# DATA
# -----------------------
DATA_CONFIG = {
    "train_path": "../data/debug_train.csv",
    "val_path": "../data/debug_val.csv",
    "vocab_path": "../data/vocab.json",
    "embeddings_dir": "../data/local_embeds/",
    "pdb_structures": "../data/local_structs/",
    "smiles_col": "rxnsmiles_ordered",
}

# -----------------------
# TRAINING
# -----------------------
def _env_int(key, default):
    """int(os.environ[key]) if set, else default. Used so a hyperparameter
    sweep can vary the decoder architecture without editing this file."""
    return int(os.environ.get(key, default))


def _env_float(key, default):
    return float(os.environ.get(key, default))


TRAIN_CONFIG = {
    "batch_size": _env_int("RXN_BATCH_SIZE", 3),
    "num_epochs": _env_int("RXN_NUM_EPOCHS", 20),
    "learning_rate": _env_float("RXN_LEARNING_RATE", 5e-5),
    "d_model": _env_int("RXN_D_MODEL", 256),
    "nhead": _env_int("RXN_NHEAD", 4),
    "num_encoder_layers": _env_int("RXN_NUM_ENCODER_LAYERS", 0),  # 0 to skip encoder
    "num_decoder_layers": _env_int("RXN_NUM_DECODER_LAYERS", 2),
    "encoder_ffn_dim": _env_int("RXN_ENCODER_FFN_DIM", 512),
    "decoder_ffn_dim": _env_int("RXN_DECODER_FFN_DIM", 512),
    "dropout": _env_float("RXN_DROPOUT", 0.1),
    "vocab_size": _env_int("RXN_VOCAB_SIZE", 8882),  # set automatically in train.py
}

# -----------------------
# MODEL
# -----------------------
MODEL_CONFIG = {
    "input_dim": int(os.environ.get("EMBED_DIM", 480)),   # ESM embedding dim (env-overridable)
    "d_model": 320,
    "nhead": 4,
    "num_encoder_layers": 0,  # set to 0 to skip encoder
    "num_decoder_layers": 2,
    "ffn_dim": 512,
    "dropout": 0.1,
}