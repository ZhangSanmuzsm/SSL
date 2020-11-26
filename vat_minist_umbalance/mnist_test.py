# 导入需要用到的包
import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from model import *
from utils import *
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

#设置一些参数，batch_size等等
labeled_batch_size = 32
unlabeled_batch_size = 64
batch_size = 32
eval_batch_size = 100
lr = 0.0003
num_epochs = 4
num_iter_per_epoch = 200
ratio = 0.7
epsilon = 0.1
# epoch_decay_start = 300
# imbalance_rate = 0.8
# cuda_device = "2"

# for imbalance_rate in [0.8,0.9,1]:
#创建数据集
train_loader = torch.utils.data.DataLoader(
      datasets.MNIST(root='./data/', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (1,))
                      ])),
      batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
      datasets.MNIST(root='./data/', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5,), (1,))
                      ])),
      batch_size=eval_batch_size, shuffle=True)

train_data = []
train_target = []

for (data, target) in train_loader:
  train_data.append(data)
  train_target.append(target)

test_data = []
test_target = []
for (data, target) in test_loader:
  test_data.append(data)
  test_target.append(target)

#把list转化为torch.tensor
train_data = torch.cat(train_data, dim=0) #torch.Size([60000, 1, 28, 28])
train_target = torch.cat(train_target, dim=0) #torch.Size([60000])
test_data = torch.cat(test_data,dim=0) #torch.Size([10000, 1, 28, 28])
test_target = torch.cat(test_target,dim=0) #torch.Size([10000])

#提取其中label是1和label是7的数据
def get_data(data,target,image_class=(1, 7),imbalance_rate=1.0,train=True):
    binary = {}
    temp_data = {}
    temp_target = {}
    for i in range(len(image_class)):
        binary[image_class[i]] = {}
        temp_data[i] = []
        temp_target[i] = []
    for i in range(len(target)):
        for j in range(len(image_class)):
            if target[i] == image_class[j]:
                temp_data[j].append(torch.unsqueeze(data[i], 0))
                temp_target[j].append(torch.unsqueeze(torch.tensor(j), 0))
    for i in range(len(image_class)):
        binary[image_class[i]]['data'] = torch.cat(temp_data[i], dim=0)
        binary[image_class[i]]['target'] = torch.cat(temp_target[i], dim=0)
    if not train:
        return binary
    sample_number = int(imbalance_rate * len(binary[image_class[0]]['target']))
    rand_sample_indices = torch.randperm(len(binary[image_class[-1]]['target']))[:sample_number]
    binary[image_class[-1]]['data'] = binary[image_class[-1]]['data'][rand_sample_indices]
    binary[image_class[-1]]['target'] = binary[image_class[-1]]['target'][rand_sample_indices]
    return binary
# 0对应image_class[0] 1对应image_class[-1]

binary_train = get_data(train_data,train_target,image_class=(1, 7),imbalance_rate=imbalance_rate,train=True)
binary_test = get_data(test_data,test_target,image_class=(1, 7),imbalance_rate=1.0,train=False)

#对每个类比按一定比例划分有标签和无标签数据
def divide_label_and_unlabel(binary_dict, image_class=(1, 7), ratio=0.7):
  sample_number_0 = int(ratio * len(binary_dict[image_class[0]]['target']))
  rand_sample_indices_0 = torch.randperm(len(binary_dict[image_class[0]]['target']))[:sample_number_0]
  indices_0 = torch.arange(len(binary_dict[image_class[0]]['target']))
  remain_sample_indices_0 = []
  for item in indices_0:
      if item not in rand_sample_indices_0:
          remain_sample_indices_0.append(torch.unsqueeze(item, 0))
  remain_sample_indices_0 = torch.cat(remain_sample_indices_0, dim=0)
  binary_dict[image_class[0]]['unlabeled_data'] = binary_dict[image_class[0]]['data'][rand_sample_indices_0]
  binary_dict[image_class[0]]['labeled_data'] = binary_dict[image_class[0]]['data'][remain_sample_indices_0]
  binary_dict[image_class[0]]['labeled_target'] = binary_dict[image_class[0]]['target'][remain_sample_indices_0]

  sample_number_1 = int(ratio * len(binary_dict[image_class[-1]]['target']))
  rand_sample_indices_1 = torch.randperm(len(binary_dict[image_class[-1]]['target']))[:sample_number_1]
  indices_1 = torch.arange(len(binary_dict[image_class[-1]]['target']))
  remain_sample_indices_1 = []
  for item in indices_1:
      if item not in rand_sample_indices_1:
          remain_sample_indices_1.append(torch.unsqueeze(item, 0))
  remain_sample_indices_1 = torch.cat(remain_sample_indices_1, dim=0)
  binary_dict[image_class[-1]]['unlabeled_data'] = binary_dict[image_class[-1]]['data'][rand_sample_indices_1]
  binary_dict[image_class[-1]]['labeled_data'] = binary_dict[image_class[-1]]['data'][remain_sample_indices_1]
  binary_dict[image_class[-1]]['labeled_target'] = binary_dict[image_class[-1]]['target'][remain_sample_indices_1]
  ### 后续可以将其改为for loop
  return binary_dict

binary_train = divide_label_and_unlabel(binary_train,image_class=(1, 7),ratio=ratio)

binary_train['labeled_data'] = torch.cat([binary_train[1]['labeled_data'],binary_train[7]['labeled_data']],dim=0)
binary_train['unlabeled_data'] = torch.cat([binary_train[1]['unlabeled_data'],binary_train[7]['unlabeled_data']],dim=0)
binary_train['labeled_target'] = torch.cat([binary_train[1]['labeled_target'],binary_train[7]['labeled_target']],dim=0)

binary_test['data'] = torch.cat([binary_test[1]['data'],binary_test[7]['data']],dim=0)
binary_test['target'] = torch.cat([binary_test[1]['target'],binary_test[7]['target']],dim=0)

# 如果使用GPU
def tocuda(x):
  return x.cuda()

#构造batch用于训练
def shuffle(images, labels=None):
  index = [i for i in range(images.shape[0])]
  random.shuffle(index)
  images = images[index]
  if labels is not None:
      labels = labels[index]
  return images, labels

def make_batch(data,target=None,batch_size=32):
  if target is not None:
      set_size = target.shape[0]
      images, labels = shuffle(data, target)
      range_list = list(range(batch_size, set_size, batch_size))
      batches = [(images[stop-batch_size: stop], labels[stop-batch_size: stop])
                for stop in range_list]
      batches.append((images[range_list[-1]: set_size], labels[range_list[-1]: set_size]))
  else:
      set_size = data.shape[0]
      images, labels = shuffle(data, target)
      range_list = list(range(batch_size, set_size, batch_size))
      batches = [images[stop-batch_size: stop] for stop in range_list]
      batches.append(images[range_list[-1]: set_size])
  return batches

labeled_batch = make_batch(binary_train['labeled_data'],binary_train['labeled_target'],batch_size=labeled_batch_size)
unlabeled_batch = make_batch(binary_train['unlabeled_data'],None,batch_size=unlabeled_batch_size)
test_batch = make_batch(binary_test['data'],binary_test['target'],batch_size=eval_batch_size)

# for (data,target) in labeled_batch:
#     print(data.shape,target.shape)  ## torch.Size([labeled_batch_size, 1, 28, 28]) torch.Size([labeled_batch_size])

#使用VAT训练
def train(model, x, y, ul_x, optimizer,alpha=0.5):

  ce = nn.CrossEntropyLoss()
  y_pred = model(x)
  ce_loss = ce(y_pred, y)

  ul_y = model(ul_x)
  v_loss = vat_loss(model, ul_x, ul_y, eps=epsilon)
  loss = alpha*v_loss + ce_loss
#     if opt.method == 'vatent':
#         loss += entropy_loss(ul_y)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  return v_loss, ce_loss


def eval(model, x, y):

  y_pred = model(x)
  prob, idx = torch.max(y_pred, dim=1)
  return torch.eq(idx, y).float().mean()


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
      m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
      m.weight.data.normal_(1.0, 0.02)
      m.bias.data.fill_(0)
  elif classname.find('Linear') != -1:
      m.bias.data.fill_(0)

model = tocuda(VAT(top_bn=False))
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=lr)

# train the network
for iters in range(num_epochs * num_iter_per_epoch):
  # if iters > epoch_decay_start * num_iter_per_epoch:
  #     decayed_lr = (num_epochs * num_iter_per_epoch - iters) * lr / (
  #                 num_epochs * num_iter_per_epoch - epoch_decay_start * num_iter_per_epoch)
  #     optimizer.lr = decayed_lr
  #     optimizer.betas = (0.5, 0.999)
  if iters == 0:
      labeled_batch = make_batch(binary_train['labeled_data'], binary_train['labeled_target'],
                                batch_size=labeled_batch_size)
      unlabeled_batch = make_batch(binary_train['unlabeled_data'], None, batch_size=unlabeled_batch_size)

  else:
      if iters % len(labeled_batch) == 0:
          labeled_batch = make_batch(binary_train['labeled_data'], binary_train['labeled_target'],
                                    batch_size=labeled_batch_size)
      if iters % len(unlabeled_batch) == 0:
          unlabeled_batch = make_batch(binary_train['unlabeled_data'], None, batch_size=unlabeled_batch_size)
  (x, y) = labeled_batch[iters % len(labeled_batch)]
  # print(x.shape)
  ul_x = unlabeled_batch[iters % len(unlabeled_batch)]
  v_loss, ce_loss = train(model.train(), Variable(tocuda(x)), Variable(tocuda(y)), Variable(tocuda(ul_x)), optimizer)

  if iters % 100 == 0:
      print("Iter :", iters, "VAT Loss :", v_loss.data, "CE Loss :", ce_loss.data)
      train_accuracy = eval(model.eval(), Variable(tocuda(x)), Variable(tocuda(y)))
      print("Labeled Train accuracy :", train_accuracy.data)
      #         train_accuracy = eval(model.eval(), Variable(tocuda(x)), Variable(tocuda(y)))
      #         print("Unlabeled Train accuracy :", train_accuracy.data)
      eval_index = torch.randperm(len(binary_test['target']))[:eval_batch_size]
      x = binary_test['data'][eval_index]
      y = binary_test['target'][eval_index]
      test_accuracy = eval(model.eval(), Variable(tocuda(x)), Variable(tocuda(y)))
      print("Test accuracy :", test_accuracy.data)

model.eval()
idx_all = []
y_all = []
for (x,y) in test_batch:
    y_pred = model(Variable(tocuda(x)))
    prob, idx = torch.max(y_pred, dim=1)
    idx_all.append(idx)
    y_all.append(y)
idx_all = torch.cat(idx_all,dim=0).cpu()
y_all = torch.cat(y_all,dim=0).cpu()
cm = confusion_matrix(y_all, idx_all)
cmp = ConfusionMatrixDisplay(cm,display_labels=[1,7])
cmp.plot(cmap=plt.cm.Blues,values_format='d')
save_dir = 'reference'
plt.title("Confusion Matrix with imbalence rate {}".format(imbalance_rate))
plt.savefig(os.path.join(save_dir, str(imbalance_rate*100)+"d100"+".pdf"))
plt.show()
# test_accuracy = 0.0
# counter = 0
# for (data, target) in test_batch:
#     n = data.size()[0]
#     acc = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
#     test_accuracy += n*acc
#     counter += n
# print("Full test accuracy :", test_accuracy.data/counter)

#根据结果画图，混淆矩阵



