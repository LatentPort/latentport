"""Validation shared by exact-token ingestion entry points."""


def token_ids(values, *, vocabulary_size=None):
    ids = tuple(values)
    if not ids or any(type(value) is not int or not 0 <= value <= 2**31 - 1 for value in ids):
        raise ValueError("Supply nonnegative integer token IDs within int32 range (not booleans)")
    if vocabulary_size is not None and any(value >= vocabulary_size for value in ids):
        raise ValueError("Token ID is outside the model vocabulary")
    return ids
