"""Validation shared by exact-token ingestion entry points."""


def token_ids(values):
    ids = tuple(values)
    if not ids or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("Supply a nonempty sequence of nonnegative integer token IDs (not booleans)")
    return ids
