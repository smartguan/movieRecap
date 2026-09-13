from __future__ import annotations
import pytest
from src.utils.hashing import hash_file, hash_string, hash_dict

def test_hash_string():
    h1 = hash_string("test")
    h2 = hash_string("test")
    assert h1 == h2
    assert h1 != hash_string("other")

def test_hash_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("content")
    h = hash_file(str(f))
    assert len(h) > 0

def test_hash_dict():
    d1 = {"a": 1, "b": [2, 3]}
    d2 = {"b": [2, 3], "a": 1}
    assert hash_dict(d1) == hash_dict(d2)
