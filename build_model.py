import torch
import Model, Embedding
from common.ModelAPI import ModelHandler, Predict,Text, ModelInfo



class Ensemble_Model(ModelHandler):
    def __init__(
        self,
        device: str = "cpu",
        model_path: str = "1000e_2025_03_08_18_06_11.pt",
    ) -> None:
        self.device = device
        self.embed_model = Embedding.get_embedding_model()
        self.model = Model.NaiveGatingModel()
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
                weights_only=True,
            )
        )
        self.model.eval()
        self.model = self.model.to(device)

    @torch.no_grad()
    def invoke(self, request: Predict.Request) -> Predict.Response:
        """使用gemini分類一個問題所屬的類別"""
        # 取出尾part
        part:Text = request.parts[-1]  
        #從part拿內容
        text = part.text
        #計算嵌入內容
        embed = self.embed_model.encode(text)
        embed = torch.from_numpy(embed).to(torch.float32).to(self.device)
        #計算閘控分數
        score = self.model(embed.unsqueeze(0)).squeeze(0)
        score = list(map(float, score)) #轉換成 Iterable[float32]
        return Predict.Response( score)

    def model_info(self) -> ModelInfo.Response:
        return ModelInfo.Response("VincentEnsemble", "1.0")



def setup() -> ModelHandler:
    return Ensemble_Model()
