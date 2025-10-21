import torchvision

trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize(512),torchvision.transforms.ToPILImage()])

train_set = torchvision.datasets.CIFAR10(root="./CIFAR10", transform=trans, train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./CIFAR10", transform=trans, train=False, download=True)

img, target = test_set[0]
img.show()

print(f"The class is:{test_set.classes[target]}")

