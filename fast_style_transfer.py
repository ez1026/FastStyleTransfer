from torchvision.models import vgg16
from torch import nn
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import cv2
import numpy


class COCODataSet(Dataset):

    def __init__(self):
        super(COCODataSet, self).__init__()
        self.zip_files = ZipFile('/mnt/d/temp/train2014.zip')
        self.data_set = []
        for file_name in self.zip_files.namelist():
            if file_name.endswith('.jpg'):
                self.data_set.append(file_name)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        file_path = self.data_set[item]
        image = self.zip_files.read(file_path)
        image = numpy.asarray(bytearray(image), dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float() / 255
        image = image.permute(2, 0, 1)
        return image


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        a = vgg16(True)
        a = a.features
        self.layer1 = a[:4]
        self.layer2 = a[4:9]
        self.layer3 = a[9:16]
        self.layer4 = a[16:23]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out1, out2, out3, out4


class ResBlock(nn.Module):

    def __init__(self, c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(c),

        )

    def forward(self, x):
        return torch.relu(self.layer(x) + x)


class TransNet(nn.Module):

    def __init__(self):
        super(TransNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 9, 1, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = nn.functional.pad(x, [10, 10, 10, 10])
        # return self.layer(x)[:,:,10:-10,10:-10]
        return self.layer(x)


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image


def get_gram_matrix(f_map):
    """
    ?????????????????????
    :param f_map:?????????
    :return:????????????????????????????????????,????????????
    """
    n, c, h, w = f_map.shape
    if n == 1:
        f_map = f_map.reshape(c, h * w)
        gram_matrix = torch.mm(f_map, f_map.t())
        return gram_matrix
    else:
        f_map = f_map.reshape(n, c, h * w)
        gram_matrix = torch.matmul(f_map, f_map.transpose(1, 2))
        return gram_matrix


if __name__ == '__main__':
    image_style = load_image('/mnt/d/temp/style.jpg').cuda()
    net = VGG16().cuda()
    g_net = TransNet().cuda()
    # summary_writer = SummaryWriter('D:/data/chapter7/logs')
    # g_net.load_state_dict(torch.load('fst.pth'))
    optimizer = torch.optim.Adam(g_net.parameters())
    loss_func = nn.MSELoss().cuda()
    data_set = COCODataSet()
    batch_size = 1
    data_loader = DataLoader(data_set, batch_size, True, drop_last=True)
    """????????????,?????????gram??????"""
    s1, s2, s3, s4 = net(image_style)
    s1 = get_gram_matrix(s1).detach()
    s2 = get_gram_matrix(s2).detach()
    s3 = get_gram_matrix(s3).detach()
    s4 = get_gram_matrix(s4).detach()
    j = 0
    count = 0
    while True:
        for i, image in enumerate(data_loader):
            """???????????????????????????"""
            image_c = image.cuda()
            image_g = g_net(image_c)
            out1, out2, out3, out4 = net(image_g)
            # loss = loss_func(image_g, image_c)
            """??????????????????"""
            loss_s1 = loss_func(get_gram_matrix(out1), s1)
            loss_s2 = loss_func(get_gram_matrix(out2), s2)
            loss_s3 = loss_func(get_gram_matrix(out3), s3)
            loss_s4 = loss_func(get_gram_matrix(out4), s4)
            loss_s = loss_s1 + loss_s2 + loss_s3 + loss_s4

            """??????????????????"""
            c1, c2, c3, c4 = net(image_c)

            # loss_c1 = loss_func(out1, c1.detach())
            loss_c2 = loss_func(out2, c2.detach())
            # loss_c3 = loss_func(out3, c3.detach())
            # loss_c4 = loss_func(out4, c4.detach())
            loss_c = loss_c2
            """?????????"""
            loss = loss_c + 0.000000005 * loss_s

            """??????????????????????????????????????????"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(j, i, loss.item())
            print(j, i, loss.item(), loss_c.item(), loss_s.item())
            count += 1
            if i % 100 == 0:
                # print(j,i, loss.item(), loss_c.item(), loss_s.item())
                torch.save(g_net.state_dict(), 'fst.pth')
                save_image([image_g[0], image_c[0]], f'/mnt/d/temp/data/{i}.jpg', padding=0, normalize=True,
                           range=(0, 1))
        j += 1

