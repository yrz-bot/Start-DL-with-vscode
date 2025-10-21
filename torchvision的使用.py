import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

trans = torchvision.transforms.Compose([torchvision.transforms.Resize(32),torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root="./CIFAR10", transform=trans, train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./CIFAR10", transform=trans, train=False, download=True)

img, target = test_set[0]
print(img.shape)

print(f"The class is:{test_set.classes[target]}")

writer = SummaryWriter("dataloader")
test_dataloader = DataLoader(dataset=test_set, batch_size=64, shuffle=False, drop_last=False,collate_fn=None)#Batch中图片尺寸要一样 所有图像转换为相同大小的张量（即通道数、高度和宽度都一致）
step = 0
for data in test_dataloader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)
    writer.add_images("batch", imgs, step)
    step += 1
writer.close()
