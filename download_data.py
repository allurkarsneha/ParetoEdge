from torchvision.datasets import EuroSAT

print("Downloading EuroSAT dataset...")

dataset = EuroSAT(
    root="./data",
    download=True
)

print("Download successful!")
print("Total images:", len(dataset))
print("Classes:", dataset.classes)