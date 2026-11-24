from mlblack.problems import SupervisedRegressionProblem


def build_problem(data):
    return SupervisedRegressionProblem(data, use_valid_objective=False)
