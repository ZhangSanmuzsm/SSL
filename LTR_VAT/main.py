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
num_valid = 100
num_iter_per_epoch = 400
eval_freq = 5
lr = 0.001



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

cuda_device = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
torch.backends.cudnn.benchmark = True

def to_var(x, requires_grad=True):
    if opt.use_cuda:
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def build_model():
    net = VAT(top_bn=False)

    if opt.use_cuda:
        net.cuda()

    optimizer = optim.Adam(net.params(), lr=lr)

    return net, optimizer

ce = nn.CrossEntropyLoss()

def train(model, x, y, ul_x, x_val, y_val, optimizer):
    model.train()

    meta_net = VAT(opt.top_bn)
    meta_net.load_state_dict(model.state_dict())
    meta_net.cuda()

    ul_x = to_var(ul_x, requires_grad=False)
    x = to_var(x, requires_grad=False)
    y = to_var(y, requires_grad=False)
    x_val = to_var(x_val, requires_grad=False)
    y_val = to_var(y_val, requires_grad=False)

    ul_y = meta_net(ul_x)
    y_g_hat = meta_net(x)

    v_loss = vat_loss(meta_net, ul_x, ul_y, eps=opt.epsilon)
    eps = to_var(torch.zeros(v_loss.size()))

    l_f_meta = torch.sum(v_loss * eps) + ce(y_g_hat, y)

    meta_net.zero_grad()
    grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
    meta_net.update_params(lr, source_params=grads)

    y_g_hat_val = meta_net(x_val)
    l_g_meta = ce(y_g_hat_val, y_val)

    grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]
    w_tilde = torch.clamp(-grad_eps, min=0)
    norm_c = torch.sum(w_tilde)
    if norm_c != 0:
        w = w_tilde / norm_c
    else:
        w = w_tilde

    y_f_hat = model(ul_x)
    y_g_hat = model(x)

    cost = vat_loss(model, ul_x, y_f_hat, eps=opt.epsilon)
    l_f = torch.sum(cost * w) + ce(y_g_hat, y)

    optimizer.zero_grad()
    l_f.backward()
    optimizer.step()

    # if opt.method == 'vatent':
    #     loss += entropy_loss(ul_y)

    return l_f, l_g_meta


def eval(model, x, y):

    y_pred = model(x)
    prob, idx = torch.max(y_pred, dim=1)
    return torch.eq(idx, y).float().mean()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('MetaConv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('MetaBatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('MetaLinear') != -1:
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
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=eval_batch_size, shuffle=True)

else:
    raise NotImplementedError

train_data = []
train_target = []

for (data, target) in train_loader:
    train_data.append(data)
    train_target.append(target)

train_data = torch.cat(train_data, dim=0)
train_target = torch.cat(train_target, dim=0)

valid_data, train_data = train_data[:num_valid, ], train_data[num_valid:, ]
valid_target, train_target = train_target[:num_valid], train_target[num_valid:, ]

labeled_train, labeled_target = train_data[:num_labeled, ], train_target[:num_labeled, ]
unlabeled_train = train_data[num_labeled:, ]

model, optimizer = build_model()
model.apply(weights_init)

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
        batch_indices_unlabeled = torch.LongTensor(np.random.choice(unlabeled_train.size()[0], unlabeled_batch_size, replace=False))
        ul_x = unlabeled_train[batch_indices_unlabeled]

        l_f, l_g_meta = train(model, x, y, ul_x, valid_data, valid_target, optimizer)

        if i % 100 == 0:
            print("Epoch :", epoch, "Iter :", i, "VAT_l_f Loss :", l_f.data, "Meta Loss :", l_g_meta.data)

    if epoch % eval_freq == 0 or epoch + 1 == opt.num_epochs:

        batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
        x = labeled_train[batch_indices]
        y = labeled_target[batch_indices]
        train_accuracy = eval(model.eval(), to_var(x, requires_grad=False), to_var(y, requires_grad=False))
        print("Train accuracy :", train_accuracy.data)

        for (data, target) in test_loader:
            test_accuracy = eval(model.eval(), to_var(data, requires_grad=False), to_var(target, requires_grad=False))
            print("Test accuracy :", test_accuracy.data)
            break


test_accuracy = 0.0
counter = 0
for (data, target) in test_loader:
    n = data.size()[0]
    acc = eval(model.eval(), to_var(data, requires_grad=False), to_var(target, requires_grad=False))
    test_accuracy += n*acc
    counter += n

print("Full test accuracy :", test_accuracy.data/counter)