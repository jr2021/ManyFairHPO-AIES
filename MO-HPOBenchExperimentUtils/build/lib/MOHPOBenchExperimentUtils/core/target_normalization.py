from typing import Union

from MOHPOBenchExperimentUtils.utils.experiment_utils import load_object


class TargetScaler:
    def __init__(self, **kwargs):
        pass

    def transform(self, y: Union[int, float]):
        raise NotImplementedError()


class NoOPScaler(TargetScaler):
    def __init__(self, **kwargs):
        super(NoOPScaler, self).__init__(**kwargs)

    def transform(self, y: Union[int, float]):
        return y


def get_scaler(algorithm: str, **scaler_parameters) -> TargetScaler:
    scaler_obj = load_object(
        import_name=algorithm,
        import_from='MOHPOBenchExperimentUtils.core.target_normalization'
    )

    scaler = scaler_obj(**scaler_parameters)
    return scaler