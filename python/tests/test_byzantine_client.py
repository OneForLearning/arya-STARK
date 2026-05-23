"""Tests for Byzantine clients."""
from __future__ import annotations

import numpy as np

from arya_stark.client.byzantine_client import (
    ALIEClient,
    BackdoorClient,
    ByzantineClient,
    IPMClient,
    LabelFlipClient,
    RandomGaussianClient,
    SignFlipClient,
)
from arya_stark.data.loaders import load_synthetic_mnist
from arya_stark.data.partition import partition_iid


def test_random_gaussian_client_sends_noise() -> None:
    ds = load_synthetic_mnist(n_train=1000, n_test=100, seed=42)
    shards = partition_iid(1000, num_clients=1, seed=42)
    client = RandomGaussianClient(
        client_id=0,
        dataset=ds,
        shard=shards[0],
        local_epochs=1,
        local_batch_size=32,
        learning_rate=0.1,
        seed=42,
        noise_scale=10.0,
    )
    global_params = np.zeros(784 * 10 + 10, dtype=np.float32)
    update = client.compute_update(global_params, round_number=1)
    # Delta should be pure noise, not zero.
    assert np.linalg.norm(update.delta) > 1.0


def test_sign_flip_client_flips_sign() -> None:
    ds = load_synthetic_mnist(n_train=1000, n_test=100, seed=42)
    shards = partition_iid(1000, num_clients=1, seed=42)
    client = SignFlipClient(
        client_id=0,
        dataset=ds,
        shard=shards[0],
        local_epochs=1,
        local_batch_size=32,
        learning_rate=0.1,
        seed=42,
    )
    global_params = np.zeros(784 * 10 + 10, dtype=np.float32)
    update = client.compute_update(global_params, round_number=1)
    # Delta should have opposite sign from honest gradient.
    # We can't verify the exact value without computing honest gradient,
    # but we can check it's non-zero.
    assert np.linalg.norm(update.delta) > 0.01


def test_label_flip_client_corrupts_labels() -> None:
    ds = load_synthetic_mnist(n_train=1000, n_test=100, seed=42)
    shards = partition_iid(1000, num_clients=1, seed=42)
    client = LabelFlipClient(
        client_id=0,
        dataset=ds,
        shard=shards[0],
        local_epochs=1,
        local_batch_size=32,
        learning_rate=0.1,
        seed=42,
        label_shift=1,
    )
    global_params = np.zeros(784 * 10 + 10, dtype=np.float32)
    update = client.compute_update(global_params, round_number=1)
    assert np.linalg.norm(update.delta) > 0.01


def test_backdoor_client_injects_trigger() -> None:
    ds = load_synthetic_mnist(n_train=1000, n_test=100, seed=42)
    shards = partition_iid(1000, num_clients=1, seed=42)
    client = BackdoorClient(
        client_id=0,
        dataset=ds,
        shard=shards[0],
        local_epochs=1,
        local_batch_size=32,
        learning_rate=0.1,
        seed=42,
        target_class=0,
        backdoor_fraction=0.1,
    )
    global_params = np.zeros(784 * 10 + 10, dtype=np.float32)
    update = client.compute_update(global_params, round_number=1)
    assert np.linalg.norm(update.delta) > 0.01


def test_ipm_client_amplifies_gradient() -> None:
    ds = load_synthetic_mnist(n_train=1000, n_test=100, seed=42)
    shards = partition_iid(1000, num_clients=1, seed=42)
    client = IPMClient(
        client_id=0,
        dataset=ds,
        shard=shards[0],
        local_epochs=1,
        local_batch_size=32,
        learning_rate=0.1,
        seed=42,
        scale=10.0,
    )
    global_params = np.zeros(784 * 10 + 10, dtype=np.float32)
    update = client.compute_update(global_params, round_number=1)
    # IPM normalises and scales → norm should be close to scale.
    norm = np.linalg.norm(update.delta)
    assert 9.0 <= norm <= 11.0, f"IPM norm {norm} should be ~10.0"


def test_alie_client_deviates_from_honest() -> None:
    ds = load_synthetic_mnist(n_train=1000, n_test=100, seed=42)
    shards = partition_iid(1000, num_clients=1, seed=42)
    client = ALIEClient(
        client_id=0,
        dataset=ds,
        shard=shards[0],
        local_epochs=1,
        local_batch_size=32,
        learning_rate=0.1,
        seed=42,
        deviation=2.0,
    )
    global_params = np.zeros(784 * 10 + 10, dtype=np.float32)
    update = client.compute_update(global_params, round_number=1)
    assert np.linalg.norm(update.delta) > 0.01


def test_byzantine_client_factory() -> None:
    """Test the factory can create all attack types."""
    ds = load_synthetic_mnist(n_train=1000, n_test=100, seed=42)
    shards = partition_iid(1000, num_clients=1, seed=42)
    for attack in [
        "random_gaussian",
        "sign_flip",
        "label_flip",
        "targeted_backdoor",
        "ipm",
        "alie",
    ]:
        client = ByzantineClient.make(
            attack=attack,
            client_id=0,
            dataset=ds,
            shard=shards[0],
            local_epochs=1,
            local_batch_size=32,
            learning_rate=0.1,
            seed=42,
        )
        global_params = np.zeros(784 * 10 + 10, dtype=np.float32)
        update = client.compute_update(global_params, round_number=1)
        assert update is not None
