import torchvision
from torch.utils.data import DataLoader

trans = torchvision.transforms.Compose([torchvision.transforms.Resize(32),torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root="./CIFAR10", transform=trans, train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./CIFAR10", transform=trans, train=False, download=True)

#img, target = test_set[0]
#img.show()

#print(f"The class is:{test_set.classes[target]}")

test_dataloader = DataLoader(dataset=test_set, batch_size=4, shuffle=False, drop_last=False,collate_fn=None)#Batch中图片尺寸要一样
for data in test_dataloader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)
