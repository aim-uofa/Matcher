r""" LVIS-92i few-shot semantic segmentation dataset """
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import cv2

from detectron2.structures.masks import *
import pycocotools.mask as mask_util

class DatasetLVIS(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 10
        self.benchmark = 'lvis'
        self.shot = shot
        self.anno_path = os.path.join(datapath, "LVIS")
        self.base_path = os.path.join(datapath, "LVIS", 'coco')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.nclass, self.class_ids_ori, self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.class_ids_c = {cid: i for i, cid in enumerate(self.class_ids_ori)}
        self.class_ids = sorted(list(self.class_ids_c.values()))

        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 2300

    def __getitem__(self, idx):
        idx %= len(self.class_ids)

        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame(idx)

        query_img = self.transform(query_img)
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(self.class_ids_c[class_sample])}

        return batch

    def build_img_metadata_classwise(self):

        with open(os.path.join(self.anno_path, 'lvis_train.pkl'), 'rb') as f:
            train_anno = pickle.load(f)
        with open(os.path.join(self.anno_path, 'lvis_val.pkl'), 'rb') as f:
            val_anno = pickle.load(f)

        train_cat_ids = list(train_anno.keys())
        val_cat_ids = [i for i in list(val_anno.keys()) if len(val_anno[i]) > self.shot]

        trn_nclass = len(train_cat_ids)
        val_nclass = len(val_cat_ids)

        nclass_val_spilt = val_nclass // self.nfolds

        class_ids_val = [val_cat_ids[self.fold + self.nfolds * v] for v in range(nclass_val_spilt)]
        class_ids_trn = [x for x in train_cat_ids if x not in class_ids_val]

        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val
        nclass = trn_nclass if self.split == 'trn' else val_nclass
        img_metadata_classwise = train_anno if self.split == 'trn' else val_anno

        return nclass, class_ids, img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata.extend(list(self.img_metadata_classwise[k].keys()))
        return sorted(list(set(img_metadata)))

    def get_mask(self, segm, image_size):

        if isinstance(segm, list):
            # polygon
            # polygons = [np.asarray(p).reshape(-1, 2)[:,::-1] for p in segm]
            # polygons = [p.reshape(-1) for p in polygons]
            polygons = [np.asarray(p) for p in segm]
            mask = polygons_to_bitmask(polygons, *image_size[::-1])
        elif isinstance(segm, dict):
            # COCO RLE
            mask = mask_util.decode(segm)
        elif isinstance(segm, np.ndarray):
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            mask = segm
        else:
            raise NotImplementedError

        return torch.tensor(mask)

    def load_frame(self, idx):

        class_sample = self.class_ids_ori[idx]

        # class_sample = np.random.choice(self.class_ids_ori, 1, replace=False)[0]
        query_name = np.random.choice(list(self.img_metadata_classwise[class_sample].keys()), 1, replace=False)[0]
        query_info = self.img_metadata_classwise[class_sample][query_name]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        org_qry_imsize = query_img.size
        query_annos = query_info['annotations']
        segms = []

        for anno in query_annos:
            segms.append(self.get_mask(anno['segmentation'], org_qry_imsize)[None, ...].float())
        query_mask = torch.cat(segms, dim=0)
        query_mask = query_mask.sum(0) > 0

        support_names = []
        support_pre_masks = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(list(self.img_metadata_classwise[class_sample].keys()), 1, replace=False)[0]
            if query_name != support_name:
                support_names.append(support_name)
                support_info = self.img_metadata_classwise[class_sample][support_name]
                support_annos = support_info['annotations']

                support_segms = []
                for anno in support_annos:
                    support_segms.append(anno['segmentation'])
                support_pre_masks.append(support_segms)

            if len(support_names) == self.shot:
                break


        support_imgs = []
        support_masks = []
        for support_name, support_pre_mask in zip(support_names, support_pre_masks):
            support_img = Image.open(os.path.join(self.base_path, support_name)).convert('RGB')
            support_imgs.append(support_img)
            org_sup_imsize = support_img.size
            sup_masks = []
            for pre_mask in support_pre_mask:
                sup_masks.append(self.get_mask(pre_mask, org_sup_imsize)[None, ...].float())
            support_mask = torch.cat(sup_masks, dim=0)
            support_mask = support_mask.sum(0) > 0

            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matcher.common import utils
    from tqdm import tqdm
    utils.fix_randseed(0)

    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    from torchvision import transforms
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    datapath = None
    use_original_imgsize = False
    fold = 0
    split = 'val'

    # cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(cls.img_mean, cls.img_std)])

    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor()])
    for fold in tqdm([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

        dataset = DatasetLVIS(datapath, fold=fold, transform=transform, split=split, shot=1,
                                          use_original_imgsize=False)

        for idx in range(len(dataset)):

            if idx > 20:
                break

            batch = dataset[idx]

            query_img, query_mask, support_imgs, support_masks = \
                batch['query_img'], batch['query_mask'], \
                batch['support_imgs'], batch['support_masks']

            imgs = torch.cat([query_img, support_imgs.squeeze()], dim=-1).permute(1,2,0).numpy()
            masks = torch.cat([query_mask, support_masks.squeeze()], dim=-1).numpy()

            query_n = batch['query_name'].split('/')[-1]

            if not os.path.exists(f'shows/lvis/fold{fold}'):
                os.makedirs(f'shows/lvis/fold{fold}')

            save_path = f'shows/lvis/fold{fold}/{query_n}'
            plt.figure(figsize=(10, 10))
            plt.imshow(imgs)
            show_mask(masks[None, ...], plt.gca())
            plt.axis('off')
            plt.savefig(save_path)