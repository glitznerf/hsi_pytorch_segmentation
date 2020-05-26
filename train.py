# Training the network

# Imports
from model import load_resnet_model
from dataset import create_dataset
import torch


# Configuration
config = {
    "file_name": "IndianPines.mat",
    "variable_name": "indian_pines",
    "augmentations": ["rot","horflip","verflip"],
    "layers": [1, 1, 1, 1],
    "pretrained": False,
    "num_classes": 16,
    "tt-split": 0.8,
    "epochs": 10,
    "batch_size": 5,
    "loss_fct": torch.nn.functional.binary_cross_entropy_with_logits,
    "optimizer": torch.optim.Adam
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Fetch data
train_data, test_data = create_dataset(config)


# Fetch model
def create_model(config):
    layers, pretrained, num_classes = config.get("layers"), config.get("pretrained"), config.get("num_classes")
    model = load_resnet_model(layers=layers, pretrained=pretrained, num_classes=num_classes)
    print(model.eval())
    print(f"Using a custom ResNet architecture with bottleneck configuration {layers}, pretrained {pretrained} and {num_classes} classes.")
    return model


# Network training
def train_model(model, config, train_data):
    optimizer = (config.get("optimizer"))(model.parameters())
    loss_fct = config.get("loss_fct")

    net = model.to(device)                                  # Move model to GPU if possible
    epochs = config.get("epochs",10)
    for epoch in range(epochs):
        net.train()
        for batch in train_data:
            x,y = (batch[0].float()).to(device), (batch[1].float()).to(device)  # Load image and ground truth batch on device
            optimizer.zero_grad()                           # Reset parameter gradients
            output = net(x)["out"]                          # Calculate output of network
            loss = loss_fct(output,y)                       # Calculate loss of output
            loss.backward()                                 # Pass loss backwards
            optimizer.step()                                # Optimize for batch
        print("Sample output: ", output[0])
        print(f"Epoch {epoch}/{epochs}, Loss: {loss}")
    return net


# Training the model
model = create_model(config)
trained_model = train_model(model, config, train_data)
