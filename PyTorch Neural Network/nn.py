import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import certifi
import ssl

# Resolve SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# GET DATA
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, batch_size=32)

# IMAGE CLASSIFIER NEURAL NETWORK
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)  # Adjust the dimensions based on the input size and convolutions
        )
    
    def forward(self, x):
        return self.model(x)

# INSTANCE OF NEURAL NETWORK, LOSS, OPTIMIZER
clf = ImageClassifier()
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# TRAINING FLOW
if __name__ == "__main__":
    # for epoch in range(10):
    #     for batch in dataset:
    #         X, y = batch
    #         yhat = clf(X)
    #         loss = loss_fn(yhat, y)  # Add target parameter

    #         # Apply backpropagation
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #     print(f"Epoch {epoch} loss is {loss.item()}")

    # with open("model_state.pt", "wb") as f:
    #     save(clf.state_dict(), f)
    with open("model_state.pt", "rb") as f:
        clf.load_state_dict(load(f))

    img = Image.open("img_2.jpg")
    img_tensor = ToTensor()(img).unsqueeze(0)

    print(torch.argmax(clf(img_tensor)))
