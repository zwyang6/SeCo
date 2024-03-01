from datasets import voc
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def build_dataloader(args):
    train_dataset = voc.VOC12ClsDataset(
    root_dir=args.data_folder,
    name_list_dir=args.list_folder,
    split=args.train_set,
    stage='train',
    aug=True,
    # resize_range=cfg.dataset.resize_range,
    rescale_range=args.scales,
    crop_size=args.crop_size,
    img_fliplr=True,
    ignore_index=args.ignore_index,
    num_classes=args.num_classes,
    )

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        #shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)

    return train_loader, train_sampler, val_loader