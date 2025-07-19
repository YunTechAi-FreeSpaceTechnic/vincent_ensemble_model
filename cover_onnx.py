
import torch
import Model, Embedding

embed_model = Embedding.get_embedding_model()
model = Model.NaiveGatingModel()

model.load_state_dict(
            torch.load(
                "1000e_2025_03_08_18_06_11.pt",
                weights_only=True,
            )
        )

embed = embed_model.encode("Hello World")
embed = torch.from_numpy(embed).to(torch.float32)

example_input = (embed.unsqueeze(0), )

onnx_program = torch.onnx.export(model, example_input, dynamo=True)

if onnx_program:
    onnx_program.save("ensemble.onnx")
