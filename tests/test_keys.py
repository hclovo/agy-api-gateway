"""Unit tests for KeyStore authentication, model scoping, usage tracking, and RPM limits."""

from __future__ import annotations

import pytest

from agy_gateway.keys import KeyAuthError, KeyStore


def test_create_list_toggle_delete_key(tmp_path) -> None:
    store = KeyStore(tmp_path / "gateway.db")
    record, raw_key = store.create_key(
        name="opencode", rpm_limit=10, allowed_models=["gemini-3.8-flash-high"]
    )
    assert raw_key.startswith("sk-agy-")
    assert record.enabled is True
    assert record.rpm_limit == 10
    assert record.allowed_models == ["gemini-3.8-flash-high"]
    assert raw_key not in record.key_prefix

    listed = store.list_keys()
    assert len(listed) == 1
    assert listed[0].id == record.id

    authed = store.authenticate(raw_key, model="gemini-3.8-flash-high")
    assert authed.id == record.id

    with pytest.raises(KeyAuthError) as exc_model:
        store.authenticate(raw_key, model="gemini-4-hydrogen-high")
    assert exc_model.value.status_code == 403

    assert store.set_enabled(record.id, False) is True
    with pytest.raises(KeyAuthError) as exc_disabled:
        store.authenticate(raw_key)
    assert exc_disabled.value.status_code == 401

    assert store.set_enabled(record.id, True) is True
    store.record_usage(record.id, input_tokens=120, output_tokens=30)
    updated = store.get_key(record.id)
    assert updated is not None
    assert updated.total_requests == 1
    assert updated.total_input_tokens == 120
    assert updated.total_output_tokens == 30
    assert updated.last_used_at is not None

    assert store.delete_key(record.id) is True
    assert store.list_keys() == []


def test_rate_limit_enforced_and_invalid_key_rejected(tmp_path) -> None:
    store = KeyStore(tmp_path / "gateway.db")
    with pytest.raises(KeyAuthError) as exc_missing:
        store.authenticate(None)
    assert exc_missing.value.status_code == 401

    with pytest.raises(KeyAuthError) as exc_invalid:
        store.authenticate("sk-agy-does-not-exist")
    assert exc_invalid.value.status_code == 401

    _, raw_key = store.create_key(name="limited", rpm_limit=2)
    store.authenticate(raw_key)
    store.authenticate(raw_key)
    with pytest.raises(KeyAuthError) as exc_rpm:
        store.authenticate(raw_key)
    assert exc_rpm.value.status_code == 429
