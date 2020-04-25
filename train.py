# Training the network

# Imports
from .model import load_resnet_model
import torch


# Configuration
config = {
    "layers": [1, 1, 1, 1],
    "pretrained": False,
    "num_classes": 1,
    "epochs": 10,
    "loss_fct": torch.nn.functional.binary_cross_entropy_with_logits,
    "optimizer": torch.optim.Adam
}


# Fetch data
#train_data=
#test_data=


# Fetch model
def create_model(config):
    model = load_resnet_model(layers=config.get("layers"), pretrained=config.get("pretrained"), num_classes=config.get("num_classes"))
    print(model.eval())
    print(f"Using a custom ResNet architecture with bottleneck configuration {layers}, pretrained {pretrained} and {num_classes} classes.")
    return model


# Model training
def train_model(model, config, train_data):
    optimizer = (config.get("optimizer"))(model.parameters())
    loss_fct = config.get("loss_fct")

    for epoch in range(config.get("epochs",10)):
        model.train()
        for batch in train_data:
            x,y = batch                         # Load image and ground truth batch
            optimizer.zero_grad()               # Reset parameter gradients
            output = model(x)                   # Calculate output of network
            loss = loss_fct(output,y)           # Calculate loss of output
            loss.backward()                     # Pass loss backwards
            optimizer.step()                    # Optimize for batch
        print(f"Loss: {loss}")
    return model


# Training the model
model = create_model(config)
trained_model = train_model(model, config, train_data)
