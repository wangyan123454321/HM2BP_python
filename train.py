import os
import time
import torch
import math
import json
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from config import reader
from data.nmnist import NMNIST
from model.spiking import Spiking
from model.counter import FiringCounter

def title(s, l = 48):
    print("")
    print("-" * l)
    print("|" + (" " * ((l - len(s)) // 2 - 1)) + s + (" " * ((l - len(s)) // 2 - 1)) + "|")
    print("-" * l)
    print("")

def print_table(values, l = 32):
    for item in values:
        l = max(len(item[0]) + 3 + 3 + 3 + 8, l)
    ll = l - 3 - 3 - 8
    print("")
    print("-" * l)
    for item in values:
        print(("|" + item[0] + (" " * (ll - len(item[0]))) + "|   " + item[1] + "|") % (item[2],))
    print("-" * l)
    print("")

def main():
    # 以下部分通过导入超参数，读取模型、数据集、优化等信息
    title("Hyper Parameters")

    config = reader.read("NMnistConfig")
    print(config.hyperparameters())

    # 以下部分构建模型
    title("Model")

    # 通过读取参数逐步构建各个层
    layers = []
    input_shape = 0
    for l in range(config.layer_count()):
        l_info = config.get_layer_by_idx(l)
        if l == 0:
            input_shape = l_info["INPUT_SHAPE"]
        if l_info["LAYER"] == "spiking":
            w_lat = None
            if "laterialType" in l_info:
                if l_info["laterialType"] == "LOCAL_INHIBITION":
                    w_lat = l_info["localInbStrength"]
            layers.append(Spiking(
                input_shape = l_info["INPUT_SHAPE"],
                output_shape = l_info["OUTPUT_SHAPE"],
                threshold = l_info["VTH"],
                tau_m = l_info["TAU_M"],
                tau_s = l_info["TAU_S"],
                t_ref = l_info["T_REFRAC"],
                weight_lateral = w_lat
            ))
    layers.append(FiringCounter())
    
    # 将各个层组合在一起成为神经网络模型
    net = torch.nn.Sequential(*layers)
    net = net.to(config.param("DEVICE"))
    print(net)

    # 以下部分加载数据集
    title("Dataset")
    
    # 训练集
    train_set = NMNIST(
        path = config.param("TRAIN_DATA_PATH"),
        train_or_test = "Train",
        data_shape = input_shape,
        time_steps = config.param("END_TIME")
    )
    # 测试集
    test_set = NMNIST(
        path = config.param("TEST_DATA_PATH"),
        train_or_test = "Test",
        data_shape = input_shape,
        time_steps = config.param("END_TIME")
    )

    sample_data, sample_label = train_set[0]
    print(sample_data.shape, sample_label) # [time_steps, input_shape], label

    # 数据加载器
    train_data_loader = DataLoader(
        dataset=train_set,
        batch_size = config.param("BATCH_SIZE"),
        shuffle = True)
    test_data_loader = DataLoader(
        dataset=test_set,
        batch_size = config.param("BATCH_SIZE"),
        shuffle=False)
    
    # 初始化环境，包括学习率衰减机制等
    title("Preparing Environment")

    loss_calc = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    # 优化器，有adam等
    optimizer = None
    if config.param("OPTIMIZER") == 'sgd':
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr = config.param("LEARNING_RATE"),
            momentum = config.param("MOMENTUM")
        )
    elif config.param("OPTIMIZER") == 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr = config.param("LEARNING_RATE"),
        )
    
    # 学习率衰减机制
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.param("TEST_EPOCH"))

    start_epoch = 0
    max_test_acc = 0

    # 将数据存在logs中，便于从tensorboard阅读数据
    out_dir = os.path.join(config.param("LOG_PATH"),
        "%d_layers_s_%d_t_%d_T_%d_b_%d_lr_%.6f_optim_%s_dataset_%s" % (
        config.layer_count(),
        input_shape,
        config.param("END_TIME"),
        config.param("TEST_EPOCH"),
        config.param("BATCH_SIZE"),
        config.param("LEARNING_RATE"),
        config.param("OPTIMIZER"),
        "nmnist"
        ))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f'Arguments and logs will be saved in {out_dir}.')

    pt_dir = out_dir + '_pt'
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)
    print(f'Model parameters will be saved in {pt_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(json.dumps(config.hyperparameters()))

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)

    # 开始训练
    title("Start Training")
    base = 0.
    offset = 1.

    # 开始训练 dot
    for epoch in range(start_epoch, config.param("TEST_EPOCH")):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        # 将标签加入偏置，正确的标签设置为35，错误的标签设置为5

        # 训练
        for image, label in tqdm(train_data_loader, desc = ("Training at epoch %d" % epoch)):
            image = image.float() # [batch_size, time_steps, input_shape]
            label_onehot = (base + offset * torch.nn.functional.one_hot(label, num_classes = config.param("NUM_CLASSES"))) # 独热编码，形状为[time_steps, batch_size, num_classes]，
            
            image = image.to(config.param("DEVICE"))
            label = label.to(config.param("DEVICE"))
            label_onehot = label_onehot.to(config.param("DEVICE"))

            image = image.permute(1, 0, 2) # [time_steps, batch_size, input_shape]
            # out = net(image) # [batch_size, num_classes]
            output_firecount = net(image) # [batch_size, num_classes]
            loss = loss_calc(output_firecount, label_onehot)

            loss.backward()
            optimizer.step()

            train_samples += label.numel()#当批次样本个数
            train_loss += loss.item() * label.numel()
            train_acc += (output_firecount.argmax(1) == label).float().sum().item()
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0


        #评估（测试） dot, time_steps, input_shape]
        with torch.no_grad():
            for image, label in tqdm(test_data_loader, desc = ("Testing at epoch %d" % epoch)):
                image = image.float() # [batch_size, time_steps, input_shape]
                label_onehot = (base + offset * torch.nn.functional.one_hot(label, num_classes = config.param("NUM_CLASSES"))) # 独热编码，形状为[time_steps, batch_size, num_classes]，
        
                image = image.to(config.param("DEVICE"))
                label = label.to(config.param("DEVICE"))
                label_onehot = label_onehot.to(config.param("DEVICE"))

                image = image.permute(1, 0, 2) # [time_steps, batch_size, input_shape]
                output_firecount = net(image) # [batch_size, num_classes]
                loss = loss_calc(output_firecount, label_onehot)
        
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (output_firecount.argmax(1) == label).float().sum().item()

        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_max.pth'))
        torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_latest.pth'))
        total_time = time.time() - start_time

        # 打印该轮次训练数据
        benchmark = [
            ["Epoch", "%8d", epoch],
            ["Training Loss", "%8.4f", train_loss],
            ["Training Accuracy", "%7.3f%%", train_acc * 100],
            ["Testing Loss", "%8.4f", test_loss],
            ["Testing Accuracy", "%7.3f%%", test_acc * 100],
            ["Maximum Testing Accuracy", "%7.3f%%", max_test_acc * 100],
            ["Duration", "%8.2f", total_time]
        ]
        print_table(benchmark)

if __name__ == '__main__':
    main()
