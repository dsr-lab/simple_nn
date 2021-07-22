import torch
import torchvision
import torchvision.transforms as transforms

from utils import DatasetLoader


class Cifar10(DatasetLoader):

    def __init__(self, batch_size=1):

        self.batch_size = batch_size

        self.train_ds = None
        self.test_ds = None

        self.train_dl = None
        self.valid_dl = None
        self.test_dl = None

        self.isDummy = False

        self.__create()

    def init_dummy_features(self):

        self.isDummy = True

        splitted_datasets = torch.utils.data.random_split(self.train_ds, [45000, 5000])
        train_subds = torch.utils.data.Subset(splitted_datasets[0], range(500))
        valid_subds = torch.utils.data.Subset(splitted_datasets[1], range(100))
        test_subds = torch.utils.data.Subset(self.test_ds, range(100))

        self.train_dl = torch.utils.data.DataLoader(train_subds, batch_size=self.batch_size, shuffle=True)
        self.valid_dl = torch.utils.data.DataLoader(valid_subds, batch_size=self.batch_size)
        self.test_dl = torch.utils.data.DataLoader(test_subds, batch_size=self.batch_size)

    def init_features(self):

        splitted_datasets = torch.utils.data.random_split(self.train_ds, [45000, 5000])

        self.train_dl = torch.utils.data.DataLoader(splitted_datasets[0], batch_size=self.batch_size, shuffle=True)
        self.valid_dl = torch.utils.data.DataLoader(splitted_datasets[1], batch_size=self.batch_size)
        self.test_dl = torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size)

    def get_full_trainin_dl(self):

        if self.isDummy:
            splitted_datasets = torch.utils.data.random_split(self.train_ds, [45000, 5000])
            train_subds = torch.utils.data.Subset(splitted_datasets[0], range(500))
            return torch.utils.data.DataLoader(train_subds, batch_size=self.batch_size, shuffle=True)

        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def __create(self, root="data"):
        self.num_classes = 10
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse',
                        'ship', 'truck')

        transformations = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda z: z.reshape(-1))])

        self.train_ds = torchvision.datasets.CIFAR10(root=root, train=True, transform=transformations,
                                                     download=True)
        self.test_ds = torchvision.datasets.CIFAR10(root=root, train=False, transform=transformations)
