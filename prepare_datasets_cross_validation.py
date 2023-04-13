import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--channels', type=int, default=3, required=False)
parser.add_argument('--height', type=int, default=385, required=False)
parser.add_argument('--width', type=int, default=385, required=False)
parser.add_argument('--folds', type=int, default=1, required=False)
args = parser.parse_args()

original_imgs = f"./data/{args.dataset}/images"
groundTruth_imgs = f"./data/{args.dataset}/manual"
borderMasks_imgs = f"./data/{args.dataset}/mask"

imgs_list = sorted(os.listdir(original_imgs))
manuals_list = sorted(os.listdir(groundTruth_imgs))
masks_list = sorted(os.listdir(borderMasks_imgs))

fold_size = len(imgs_list)//args.folds

folds_imgs_list = [imgs_list[fold_idx*fold_size:fold_idx*fold_size+fold_size] for fold_idx in range(args.folds)]
folds_manuals_list = [manuals_list[fold_idx*fold_size:fold_idx*fold_size+fold_size] for fold_idx in range(args.folds)]
folds_masks_list = [masks_list[fold_idx*fold_size:fold_idx*fold_size+fold_size] for fold_idx in range(args.folds)]

test_Nimgs = fold_size
train_Nimgs = len(imgs_list)-test_Nimgs

for cross_validation_idx in range(args.folds):
    cross_validation_dataset = f"{args.dataset}_cross_validation_{cross_validation_idx}"

    if os.path.isdir(f"./data/{cross_validation_dataset}"):
        shutil.rmtree(f"./data/{cross_validation_dataset}")

    os.mkdir(f"./data/{cross_validation_dataset}")
    os.mkdir(f"./data/{cross_validation_dataset}/training")
    os.mkdir(f"./data/{cross_validation_dataset}/test")

    fold_original_imgs_train = f"./data/{cross_validation_dataset}/training/images"
    fold_groundTruth_imgs_train = f"./data/{cross_validation_dataset}/training/manual"
    fold_borderMasks_imgs_train = f"./data/{cross_validation_dataset}/training/mask"
    fold_original_imgs_test = f"./data/{cross_validation_dataset}/test/images"
    fold_groundTruth_imgs_test = f"./data/{cross_validation_dataset}/test/manual"
    fold_borderMasks_imgs_test = f"./data/{cross_validation_dataset}/test/mask"

    os.mkdir(fold_original_imgs_train)
    os.mkdir(fold_groundTruth_imgs_train)
    os.mkdir(fold_borderMasks_imgs_train)
    os.mkdir(fold_original_imgs_test)
    os.mkdir(fold_groundTruth_imgs_test)
    os.mkdir(fold_borderMasks_imgs_test)

    for fold_idx in range(args.folds):
        fold_img_list = folds_imgs_list[fold_idx]
        fold_manuals_list = folds_manuals_list[fold_idx]
        fold_masks_list = folds_masks_list[fold_idx]

        for img_name in fold_img_list:
            if cross_validation_idx==fold_idx:
                src_img = os.path.join(original_imgs, img_name)
                dst_img = os.path.join(fold_original_imgs_test, img_name)
            else:
                src_img = os.path.join(original_imgs, img_name)
                dst_img = os.path.join(fold_original_imgs_train, img_name)
            shutil.copy(src_img, dst_img)

        for manual_name in fold_manuals_list:
            if cross_validation_idx==fold_idx:
                src_manual = os.path.join(groundTruth_imgs, manual_name)
                dst_manual = os.path.join(fold_groundTruth_imgs_test, manual_name)
            else:
                src_manual = os.path.join(groundTruth_imgs, manual_name)
                dst_manual = os.path.join(fold_groundTruth_imgs_train, manual_name)
            shutil.copy(src_manual, dst_manual)

        for mask_name in fold_masks_list:
            if cross_validation_idx==fold_idx:
                src_mask = os.path.join(borderMasks_imgs, mask_name)
                dst_mask = os.path.join(fold_borderMasks_imgs_test, mask_name)
            else:
                src_mask = os.path.join(borderMasks_imgs, mask_name)
                dst_mask = os.path.join(fold_borderMasks_imgs_train, mask_name)
            shutil.copy(src_mask, dst_mask)


    os.system((
        f"python prepare_datasets.py "
        f"--dataset {cross_validation_dataset} "
        f"--train_Nimgs {train_Nimgs} "
        f"--test_Nimgs {test_Nimgs} "
        f"--channels {args.channels} "
        f"--height {args.height} "
        f"--width {args.width} "
        f"--prep_config True" 
    ))