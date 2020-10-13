import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from model import *
from utils import *
import os

batch_size = 32
eval_batch_size = 100
unlabeled_batch_size = 128
num_labeled = 1000
num_valid = 0
num_iter_per_epoch = 400
eval_freq = 5
lr = 0.001
cuda_device = "2"
# torch.backends.cudnn.enabled = False


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=120)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--epsilon', type=float, default=2.5)
parser.add_argument('--top_bn', type=bool, default=True)
parser.add_argument('--method', default='vat')


opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def train(model, x, y, ul_x, optimizer):

    ce = nn.CrossEntropyLoss()
    y_pred = model(x)
    ce_loss = ce(y_pred, y)

    ul_y = model(ul_x)
    v_loss = vat_loss(model, ul_x, ul_y, eps=opt.epsilon)
    loss = v_loss + ce_loss
    if opt.method == 'vatent':
        loss += entropy_loss(ul_y)

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


if opt.dataset == 'svhn':
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='train', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='test', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=eval_batch_size, shuffle=True)

elif opt.dataset == 'cifar10':
    num_labeled = 4000
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
                      ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
                      ])),
        batch_size=eval_batch_size, shuffle=True)

else:
    raise NotImplementedError

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

train_data = torch.cat(train_data, dim=0)
train_target = torch.cat(train_target, dim=0)
test_data = torch.cat(test_data,dim=0)
test_target = torch.cat(test_target,dim=0)

train_data = train_data.reshape((-1,3*32*32)).numpy()
test_data = test_data.reshape((-1,3*32*32)).numpy()
print("Apply ZCA whitening")
components, mean, train_data = ZCA(train_data)
np.save('{}/components'.format('.'), components)
np.save('{}/mean'.format('.'), mean)
test_data = np.dot(test_data - mean, components.T)
train_data = torch.from_numpy(train_data.reshape((-1,3,32,32)))
test_data = torch.from_numpy(test_data.reshape((-1,3,32,32)))

print(train_data.shape)
print(test_data.shape)



valid_data, train_data = train_data[:num_valid, ], train_data[num_valid:, ]
valid_target, train_target = train_target[:num_valid], train_target[num_valid:, ]

labeled_train, labeled_target = train_data[:num_labeled, ], train_target[:num_labeled, ]
#unlabeled_train+train = train_data

model = tocuda(VAT(opt.top_bn))
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=lr)

# print(labeled_train.shape) # (4000,3,32,32)
# print(labeled_train[0])
# print(labeled_target.shape) # (4000)


# train the network
for epoch in range(opt.num_epochs):

    if epoch > opt.epoch_decay_start:
        decayed_lr = (opt.num_epochs - epoch) * lr / (opt.num_epochs - opt.epoch_decay_start)
        optimizer.lr = decayed_lr
        optimizer.betas = (0.5, 0.999)

    for i in range(num_iter_per_epoch):

        batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
        x = labeled_train[batch_indices]
        y = labeled_target[batch_indices]
        batch_indices_unlabeled = torch.LongTensor(np.random.choice(train_data.size()[0], unlabeled_batch_size, replace=False))
        ul_x = train_data[batch_indices_unlabeled]

        v_loss, ce_loss = train(model.train(), Variable(tocuda(x)), Variable(tocuda(y)), Variable(tocuda(ul_x)),
                                optimizer)

        if i % 100 == 0:
            print("Epoch :", epoch, "Iter :", i, "VAT Loss :", v_loss.data, "CE Loss :", ce_loss.data)

    if epoch % eval_freq == 0 or epoch + 1 == opt.num_epochs:

        batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
        x = labeled_train[batch_indices]
        y = labeled_target[batch_indices]
        train_accuracy = eval(model.eval(), Variable(tocuda(x)), Variable(tocuda(y)))
        print("Train accuracy :", train_accuracy.data)

        eval_batch_indices = torch.LongTensor(np.random.choice(test_data.size()[0], eval_batch_size, replace=False))
        x = test_data[eval_batch_indices]
        y = test_target[eval_batch_indices]
        test_accuracy = eval(model.eval(), Variable(tocuda(x)), Variable(tocuda(y)))
        print("Test accuracy :", test_accuracy.data)


test_accuracy = 0.0
counter = 0
for (data, target) in zip(test_data,test_target):
    n = data.size()[0]
    acc = eval(model.eval(), Variable(tocuda(data.unsqueeze(0))), Variable(tocuda(target)))
    test_accuracy += n*acc
    counter += n

print("Full test accuracy :", test_accuracy.data/counter)