from operator import index
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment


def auto_classify(data, target):
    experiment = ClassificationExperiment()
    experiment.setup(data=data, target=target)
    setup_df = experiment.pull()
    best_model = experiment.compare_models()
    compare_df = experiment.pull()
    experiment.save_model(best_model, "_model")

    return setup_df, compare_df

def auto_regression(data, target):
    experiment = RegressionExperiment()
    experiment.setup(data=data, target=target)
    setup_df = experiment.pull()
    best_model = experiment.compare_models()
    compare_df = experiment.pull()
    experiment.save_model(best_model, "_model")

    return setup_df, compare_df

