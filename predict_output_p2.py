import argparse
from PIL import Image
import torch
import glob
import os
from torch.utils.data import DataLoader,Dataset
import numpy as np
import viz_mask
import torchvision.transforms as transforms
import imageio
import torch.nn as nn
import torchvision.models as models
import mean_iou_evaluate


class FCN8(nn.Module):
    def __init__(self):
        super(FCN8, self).__init__()
        self.vgg_feature = models.vgg16(pretrained=True).features

        self.p0_p3 = self.vgg_feature[:17]
        self.p3_p4 = self.vgg_feature[17:24]
        self.p4_p5 = self.vgg_feature[24:31]
        
        #upsample 2x
        self.up2x = nn.Upsample(scale_factor= 2 , mode='bilinear', align_corners=True)
        #upsample 8x
        self.ct8x = nn.Sequential(
            nn.ConvTranspose2d(256,7,8,8),
            nn.ReLU()
        )
        #shape channel from 512 to 256
        self.ct = nn.Sequential(
            nn.ConvTranspose2d(512,256,2,stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        p3 = self.p0_p3(x)
        p4 = self.p3_p4(p3)
        p5 = self.p4_p5(p4)

        p4p5 = self.up2x(p5) + p4
        p4p5 = self.ct(p4p5)
        p3p4p5 = p4p5+p3

        x = self.ct8x(p3p4p5)
        return x

class PRED(Dataset):
    def __init__(self, root):
        self.X = None
        self.filenames = []
        self.filepaths =  glob.glob(os.path.join(root,'*.jpg'))
        self.images = []
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])

        for i in self.filepaths :
            self.filenames.append(os.path.splitext(os.path.basename(i))[0])
            # self.images.append(imageio.imread(i))
        
        self.len = len(self.filenames)
    
    def __getitem__(self, index) :
        filepath = self.filepaths[index]
        filename = self.filenames[index]
        # img = self.images[index]
        X = Image.open(filepath)
        X = self.transform(X)
        # img = self.transform(img)
        #img(transformed), filename, imgio
        # return X,filename,img
        return X,filename

    def __len__(self):
        return self.len

def output_to_png(filename, masks, out_path):

    pass

            
    
def test_and_output(model,test_dataloader, out_path):
    print('test_and_output')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i, (data, filename) in enumerate(test_dataloader):
        data = data.to(device)
        output = model(data)
        pred = output.max(1, keepdim=False)[1].cpu().numpy()
        for j in range(len(pred)):
            output_to_png(filename[i],pred[i],out_path)
        
        

if __name__=='__main__':

    # parser =  argparse.ArgumentParser(description='Use to predict image for HW1_p1')
    # parser.add_argument( '--img_path', type=str, default='', help='path to image' )
    # parser.add_argument( '--out_path', type=str, default='', help='path to out')
    # args = parser.parse_args()

    img_path = './HW1/p2_data/validation'
    out_path = './HW1/p2_data/out'
    # model = torch.load('hw1_p1.pth')

    test_dataset = PRED(img_path)
    test_dataloader = DataLoader(test_dataset, batch_size=2)
    model = torch.load('./HW1/done.pth')
    test_and_output(model, test_dataloader, out_path)