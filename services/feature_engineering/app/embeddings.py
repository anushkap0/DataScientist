"""
Optional Hugging Face text-embedding step. Free-text columns (product
reviews, comments, descriptions — as opposed to short categorical labels)
carry real signal that label-encoding throws away, so we embed them with a
small sentence-transformer instead and feed the dense vector into the same
sklearn models everything else uses.

Model loads lazily and once per process (it's ~90MB, a few seconds to
download on first call, cached afterward). If sentence-transformers/torch
aren't installed or the model can't be fetched (e.g. no network), we log a
warning and the caller falls back to label encoding — this must never be
the reason a pipeline run fails.
"""
import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger("feature_engineering.embeddings")

HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TEXT_EMBEDDING_COMPONENTS = int(os.getenv("TEXT_EMBEDDING_COMPONENTS", "16"))
FREE_TEXT_MIN_AVG_WORDS = 4  # below this, treat as a categorical label, not free text

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading Hugging Face embedding model %s", HF_EMBEDDING_MODEL)
        _model = SentenceTransformer(HF_EMBEDDING_MODEL)
    return _model


def is_free_text_column(series: pd.Series) -> bool:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    avg_words = non_null.str.split().str.len().mean()
    return avg_words >= FREE_TEXT_MIN_AVG_WORDS


def embed_column(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Returns a DataFrame of dense embedding columns, reduced to
    TEXT_EMBEDDING_COMPONENTS dims via SVD so one text column doesn't
    dwarf the rest of the feature space."""
    texts = series.fillna("").astype(str).tolist()
    model = _get_model()  # raises if unavailable — caller catches this
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    n_components = min(TEXT_EMBEDDING_COMPONENTS, vectors.shape[1], max(vectors.shape[0] - 1, 1))
    if n_components < vectors.shape[1]:
        from sklearn.decomposition import TruncatedSVD
        vectors = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(vectors)

    cols = [f"{prefix}_emb_{i}" for i in range(vectors.shape[1])]
    return pd.DataFrame(vectors, columns=cols, index=series.index)
