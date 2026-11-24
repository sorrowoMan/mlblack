from mlblack.integrations.nsgablack_optimization import build_optimization_adapter


def build_adapter(*, learning_rate):
    return build_optimization_adapter("gradient.sgd", learning_rate=float(learning_rate))
