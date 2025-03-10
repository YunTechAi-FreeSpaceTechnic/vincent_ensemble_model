from common.ModelAPI import ModelHandler, Predict, ModelInfo

class TemplateModel(ModelHandler):
    def invoke(self, request: Predict.Request):
        return Predict.Response([])

    def model_info(self) -> ModelInfo.Response:
        return ModelInfo.Response("Template", "1.0")


def setup() -> ModelHandler:
    return TemplateModel()
