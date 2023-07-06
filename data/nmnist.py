import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import re

class NMNIST(Dataset):
    def __init__(self, path = "./data/", train_or_test = "Train", data_shape = 2312, time_steps = 540) -> None:    
        super().__init__()
        self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        extract_path = path + os.sep + "extract_" + train_or_test + "_" + str(data_shape) + "_" + str(time_steps)
        if not os.path.isdir(extract_path):
            os.mkdir(extract_path)
        end_flag = extract_path + "_end_flag"
        if not os.path.isfile(end_flag):
            for c in self.classes:
                class_file = path + os.sep + train_or_test + "_" + c + ".dat"
                print("Extracting file %s" % (class_file,))
                with open(class_file, "r") as f:
                    img_idx = 0
                    current_spike_train = np.zeros((time_steps, data_shape), dtype = np.int8)
                    l = f.readline()
                    while l:
                        if l[0] == "#":
                            img_idx += 1
                            save_file_name = c + "_" + str(img_idx) + ".npz"
                            np.savez(extract_path + os.sep + save_file_name, data = current_spike_train, label = self.classes.index(c))
                            print("successfully processed %s" % (extract_path + os.sep + save_file_name,))
                            current_spike_train = np.zeros((time_steps, data_shape), dtype = np.int8)
                            l = f.readline()
                            continue
                        info = re.sub(r"[^0-9]+$", "", re.sub(r"[^0-9\t]", "", l)).split("\t")
                        info = np.array([int(v) for v in info])
                        s_index = info[0] - 1
                        t_indices = info[1:] - 1
                        qualified_indices = np.where(t_indices < time_steps)
                        t_indices = t_indices[qualified_indices]
                        current_spike_train[t_indices, s_index] = 1
                        l = f.readline()
            with open(end_flag, "w") as f:
                f.write("1")
        self.file_path = extract_path
        self.data_files = os.listdir(extract_path)
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, index):
        data_and_label = np.load(self.file_path + os.sep + self.data_files[index])
        return data_and_label["data"], data_and_label["label"]