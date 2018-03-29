import torchvision.transforms as transforms

from datasets import AVADataset, TestDataset
from torch.utils.data import DataLoader


def get_data_loader(opt):
    if opt.is_train:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        dataset = AVADataset(csv_file=opt.train_csv_file, root_dir=opt.img_path, transform=transform)
        batch_size = opt.train_batch_size
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor()
            ])
        dataset = TestDataset(image_dir=opt.test_image_dir, root_dir=opt.test_path, transform=transform)
        batch_size = 1

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=opt.is_train > 0,
        num_workers=opt.num_workers)


def get_val_data_loader(opt):
    if not opt.is_validation:
        return None

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor()])
    dataset = AVADataset(csv_file=opt.val_csv_file, root_dir=opt.img_path, transform=transform)
    batch_size = opt.val_batch_size

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=opt.num_workers)
