r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from .coco import DatasetCOCO
from .pascal import DatasetPASCAL
from .fss import DatasetFSS
from .paco_part import DatasetPACOPart
from .pascal_part import DatasetPASCALPart
from .lvis import DatasetLVIS

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            'coco': DatasetCOCO,
            'pascal': DatasetPASCAL,
            'fss': DatasetFSS,
            'paco_part': DatasetPACOPart,
            'pascal_part': DatasetPASCALPart,
            'lvis': DatasetLVIS,
        }

        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader
