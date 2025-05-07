import os
import nibabel as nib
import numpy as np
import torch

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.test_flag = test_flag
        if test_flag:
            self.seqtypes = ["voided", "mask"]
        else:
            self.seqtypes = ["voided", "mask", "t1n"]

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        self.mask_vis = []

        for root, dirs, _ in os.walk(self.directory):
            dirs_sorted = sorted(dirs)
            for dir_id in dirs_sorted:
                datapoint = dict()
                sli_dict = dict()
                dir_path = os.path.join(root, dir_id)
                for _, _, files in os.walk(dir_path):
                    files_sorted = sorted(files)
                    for f in files_sorted:
                        seqtype = f.split("-")[-1].split(".")[0]
                        if seqtype in self.seqtypes_set:
                            datapoint[seqtype] = os.path.join(dir_path, f)

                    if "mask" in datapoint:
                        slice_range = []
                        mask_array = np.array(nib.load(datapoint["mask"]).dataobj)
                        if test_flag:
                            mask_array = np.pad(mask_array, ((0, 0), (0, 0), (34, 35)))
                            mask_array = mask_array[8:-8, 8:-8, :]
                        for i in range(0, 224):
                            mask_slice = mask_array[:, :, i]
                            if np.sum(mask_slice) != 0:
                                slice_range.append(i)

                        self.database.append(datapoint)
                        self.mask_vis.append(slice_range)

            break  # prevent descending into sub-subdirectories

    def __getitem__(self, x):
        filedict = self.database[x]
        slicedict = self.mask_vis[x]
        out_single = []

        if self.test_flag:
            for seqtype in self.seqtypes:
                nib_img = np.array(nib.load(filedict[seqtype]).dataobj).astype(np.float32)
                if seqtype == "voided":
                    nib_img = np.pad(nib_img, ((0, 0), (0, 0), (34, 35)))
                    nib_img = np.clip(nib_img, np.quantile(nib_img, 0.001), np.quantile(nib_img, 0.999))
                    nib_img = (nib_img - np.min(nib_img)) / (np.max(nib_img) - np.min(nib_img))
                else:  # mask
                    nib_img = np.pad(nib_img, ((0, 0), (0, 0), (34, 35)))
                out_single.append(torch.tensor(nib_img))

            out_single = torch.stack(out_single)
            image = out_single[0:2, ...]
            path = filedict["voided"]
            return (image, path, slicedict)

        else:
            for seqtype in self.seqtypes:
                nib_img = np.array(nib.load(filedict[seqtype]).dataobj).astype(np.float32)
                out_single.append(torch.tensor(nib_img))

            out_single = torch.stack(out_single)
            image = out_single[0:2, ...]
            label = out_single[2, ...].unsqueeze(0)
            path = filedict["t1n"]
            return (image, label, path, slicedict)

    def __len__(self):
        return len(self.database)
