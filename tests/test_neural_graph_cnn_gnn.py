from __future__ import annotations

import numpy as np

from mlblack.core import ComputeBackendSession, ComputeBackendSpec
from mlblack.pipeline.data import GraphDataView, ImageContrastivePairDataView, ImageDataView
from mlblack.presets import (
    build_tiny_cnn_image_classification_trainer,
    build_tiny_cnn_image_contrastive_trainer,
    build_tiny_gnn_graph_classification_trainer,
)
from mlblack.representations.codecs import NeuralGraphCodec, NeuralGraphSpec


def _backend_context() -> dict[str, object]:
    return {"backend.session": ComputeBackendSession(ComputeBackendSpec(name="torch", device="cpu"))}


def test_tiny_cnn_codec_and_image_classification_trainer_smoke() -> None:
    spec = NeuralGraphSpec.tiny_cnn(
        channels=1,
        height=4,
        width=4,
        conv_channels=(4,),
        heads=({"kind": "classification", "name": "classification", "params": {"num_classes": 2}},),
    )
    codec = NeuralGraphCodec(spec, random_seed=31)
    ctx = _backend_context()
    model = codec.decode(codec.init_values(ctx), ctx)
    assert model.describe()["kind"] == "tiny_cnn"

    data = _image_data()
    trainer = build_tiny_cnn_image_classification_trainer(
        data,
        conv_channels=(4,),
        num_classes=2,
        learning_rate=1e-2,
        random_seed=31,
    )
    result = trainer.fit(max_steps=2)
    assert result.best_feedback is not None
    assert "train.accuracy" in result.best_feedback.metrics
    assert result.report["adapter"]["name"] == "neural_graph_backprop"
    assert result.report["problem"]["route"] == "tiny_cnn"


def test_tiny_gnn_codec_and_graph_classification_trainer_smoke() -> None:
    spec = NeuralGraphSpec.tiny_gnn(
        node_feature_dim=3,
        num_nodes=4,
        hidden_dim=6,
        num_layers=2,
        heads=({"kind": "classification", "name": "classification", "params": {"num_classes": 2}},),
    )
    codec = NeuralGraphCodec(spec, random_seed=37)
    ctx = _backend_context()
    model = codec.decode(codec.init_values(ctx), ctx)
    assert model.describe()["kind"] == "tiny_gnn"

    data = _graph_data()
    trainer = build_tiny_gnn_graph_classification_trainer(
        data,
        hidden_dim=6,
        num_layers=2,
        num_classes=2,
        learning_rate=1e-2,
        random_seed=37,
    )
    result = trainer.fit(max_steps=2)
    assert result.best_feedback is not None
    assert "train.accuracy" in result.best_feedback.metrics
    assert result.report["adapter"]["name"] == "neural_graph_backprop"
    assert result.report["problem"]["route"] == "tiny_gnn"


def test_tiny_cnn_contrastive_retrieval_trainer_smoke() -> None:
    data = _image_pair_data()
    trainer = build_tiny_cnn_image_contrastive_trainer(
        data,
        conv_channels=(4,),
        embedding_dim=4,
        learning_rate=1e-2,
        random_seed=41,
    )
    result = trainer.fit(max_steps=2)
    assert result.best_feedback is not None
    assert "train.retrieval_accuracy" in result.best_feedback.metrics
    assert result.report["problem"]["head"] == "retrieval"


def _image_data() -> ImageDataView:
    X = np.zeros((6, 1, 4, 4), dtype=float)
    X[:3, :, :2, :2] = 1.0
    X[3:, :, 2:, 2:] = 1.0
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
    return ImageDataView(X_train=X, y_train=y)


def _graph_data() -> GraphDataView:
    node_features = np.zeros((6, 4, 3), dtype=float)
    adjacency = np.zeros((6, 4, 4), dtype=float)
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
    for idx in range(6):
        adjacency[idx] = np.eye(4)
        if idx < 3:
            adjacency[idx, 0, 1] = adjacency[idx, 1, 0] = 1.0
            node_features[idx, :, 0] = 1.0
        else:
            adjacency[idx, 2, 3] = adjacency[idx, 3, 2] = 1.0
            node_features[idx, :, 1] = 1.0
    return GraphDataView(node_features_train=node_features, adjacency_train=adjacency, y_train=y)


def _image_pair_data() -> ImageContrastivePairDataView:
    anchors = np.zeros((4, 1, 4, 4), dtype=float)
    positives = np.zeros_like(anchors)
    negatives = np.zeros_like(anchors)
    anchors[:, :, :2, :2] = 1.0
    positives[:, :, :2, :2] = 0.9
    negatives[:, :, 2:, 2:] = 1.0
    return ImageContrastivePairDataView(anchor_train=anchors, positive_train=positives, negative_train=negatives)
