"""Verified in-memory encoder state and equivalent body-ID fast/fallback paths."""

import json
import pickle
from copy import deepcopy
from types import SimpleNamespace

import dill
import numpy as np
import pytest
from data_pipeline_pretrain.pipeline import root_data as rd
from datatrove.data import Document
from datatrove.executor.base import ExecutorJSONEncoder
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from tokenization_scripts.tokenize import EncodeTokens, LocalBodyEncoder


def encoder():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "café": 1, "你好": 2}, "[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["<control>"])
    return LocalBodyEncoder(tokenizer.to_str().encode())


def spec():
    return rd.TokenSpec(
        encoding={"fixture": "unicode"},
        vocabulary_size=4,
        prefix_tokens=(3,),
        suffix_tokens=(3,),
    )


def test_fast_encoding_and_regular_fallback_keep_body_ids():
    body = encoder()
    texts = ["café 你好", "", "unseen", "<control>"]
    expected = [
        value.ids
        for value in body.tokenizer.encode_batch(texts, add_special_tokens=False)
    ]
    assert expected[-1] == [3]
    assert body(tuple(texts)) == expected
    tokenizer = body.tokenizer
    # Exercise the supported tokenizers API without encode_batch_fast.
    body._tokenizer = SimpleNamespace(encode_batch=tokenizer.encode_batch)
    rows = body(texts)
    assert rows == expected


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("transport", ["pickle", "dill", "deepcopy"])
@pytest.mark.parametrize("pipeline", [False, True])
def test_encoder_transfers_only_verified_bytes_and_reuses_local_model(
    warm, transport, pipeline
):
    body = encoder()
    step = EncodeTokens(body.asset, spec(), batch_size=2, batch_bytes=17)
    if pipeline:
        body = step.encoder
    expected = [[1, 2]]
    if warm:
        assert body(["café 你好"]) == expected
        assert body.tokenizer is body.tokenizer
    state = body.__getstate__()
    assert state == {"asset": body.asset}
    target = step if pipeline else body
    if transport == "deepcopy":
        restored = deepcopy(target)
    elif transport == "dill":
        restored = dill.loads(dill.dumps(target))
    else:
        restored = pickle.loads(pickle.dumps(target))
    restored_body = restored.encoder if pipeline else restored
    assert restored_body._tokenizer is None
    if pipeline:
        assert (restored.batch_size, restored.batch_bytes) == (2, 17)
        document = Document(text="café 你好", id="duplicate")
        assert list(restored([document], 0, 1)) == [document]
        assert np.frombuffer(document.metadata["tokens"], dtype="<i4").tolist() == [
            3,
            1,
            2,
            3,
        ]
        assert document.metadata["stored_tokens"] == 4
        restored = restored.encoder
    assert restored.asset == body.asset
    assert restored(["café 你好"]) == expected
    assert restored.tokenizer is restored.tokenizer
    logged = json.dumps(target, cls=ExecutorJSONEncoder)
    assert body.asset.decode() not in logged
    if pipeline:
        settings = json.loads(logged)
        assert settings["encoder"] == repr(body)
        assert (settings["batch_size"], settings["batch_bytes"]) == (2, 17)
    else:
        assert json.loads(logged) == repr(body)


@pytest.mark.parametrize(
    "rows,byte_limit,expected_batches",
    [
        (2, 64, [["café", "你好"], ["a", "xxxxxxxxx"], ["b", "c"]]),
        (100, 7, [["café"], ["你好", "a"], ["xxxxxxxxx"], ["b", "c"]]),
    ],
)
def test_body_step_bounds_input_and_preserves_documents(
    monkeypatch, rows, byte_limit, expected_batches
):
    step = EncodeTokens(
        encoder().asset, spec(), batch_size=rows, batch_bytes=byte_limit
    )
    texts = ["café", "你好", "a", "xxxxxxxxx", "b", "c"]
    documents = [
        Document(
            text=text,
            id="duplicate",
            metadata={"root_position": ("data/text@v1", index), "kept": index},
        )
        for index, text in enumerate(texts)
    ]
    calls = []
    encode = LocalBodyEncoder.__call__

    def observed(self, texts):
        calls.append(texts)
        return encode(self, texts)

    monkeypatch.setattr(LocalBodyEncoder, "__call__", observed)
    result = list(step(iter(documents), 1, 2))
    assert calls == expected_batches
    assert all(actual is original for actual, original in zip(result, documents))
    assert len(result) == len(documents)
    for index, document in enumerate(result):
        assert document.text == texts[index] and document.id == "duplicate"
        expected = 1 if index == 0 else 2 if index == 1 else 0
        assert np.frombuffer(document.metadata["tokens"], dtype="<i4").tolist() == [
            3,
            expected,
            3,
        ]
        assert document.metadata["stored_tokens"] == 3
        assert document.metadata["root_position"] == ("data/text@v1", index)
        assert document.metadata["kept"] == index


@pytest.mark.parametrize("count", [0, 1, 3])
def test_body_step_rejects_encoder_cardinality_before_yield(monkeypatch, count):
    step = EncodeTokens(encoder().asset, spec())
    documents = [Document(text="café", id=str(index)) for index in range(2)]
    monkeypatch.setattr(LocalBodyEncoder, "__call__", lambda self, texts: [[1]] * count)
    with pytest.raises(ValueError, match="exactly one body"):
        next(step(documents))
    assert all("tokens" not in document.metadata for document in documents)


def test_body_step_empty_partition_remains_lazy():
    step = EncodeTokens(encoder().asset, spec())
    assert list(step([])) == []
    assert step.encoder._tokenizer is None


def test_malformed_asset_keeps_lazy_failure():
    body = LocalBodyEncoder(b"{")
    with pytest.raises(Exception, match="EOF while parsing"):
        body(["café"])


def test_empty_body_preserves_rows_with_boundaries(monkeypatch):
    step = EncodeTokens(encoder().asset, spec())
    documents = [Document(text="café", id="0"), Document(text="", id="1")]
    monkeypatch.setattr(LocalBodyEncoder, "__call__", lambda self, texts: [[1], []])
    assert list(step(documents)) == documents
    assert np.frombuffer(documents[1].metadata["tokens"], dtype="<i4").tolist() == [
        3,
        3,
    ]
    assert documents[1].metadata["stored_tokens"] == 2


def test_control_ids_in_text_are_preserved_with_one_explicit_boundary():
    step = EncodeTokens(encoder().asset, spec())
    document = Document(text="<control>", id="0")
    assert list(step([document])) == [document]
    assert np.frombuffer(document.metadata["tokens"], dtype="<i4").tolist() == [3, 3, 3]
    assert document.metadata["stored_tokens"] == 3
