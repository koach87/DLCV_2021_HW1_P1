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


class PRED(Dataset):
    def __init__(self, root):
        self.X = None
        self.filenames = []
        self.filepaths =  glob.glob(os.path.join(root,'*.jpg'))
        self.root = root
        self.transform = transforms.Compose(
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        )

        for i in self.filepaths :
            self.filenames.append(os.path.splitext(os.path.basename(i))[0])
        
        self.len = len(self.filenames)
    
    def __getitem__(self, index) :
        filepath = self.filepaths[index]
        self.X = Image.open(filepath)
        return self.X

    def __len__(self):
        return self.len


def arg_parse():
    parser =  argparse.ArgumentParser(description='Use to predict image for HW1_p1')
    parser.add_argument(
        '--img_path',
        type=str,
        default='',
        help='path to load'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default='',
        help='path to save'
    )
    args = parser.parse_args()
    return args

def output_to_png(filenames, masks, img, out_path):
    for filename in filenames:
        cs = np.unique(masks)
        for c in cs:
            mask = np.zeros((img.shape[0], img.shape[1]))
            ind = np.where(masks==c)
            mask[ind[0], ind[1]] = 1
            img =  viz_mask.viz_data(img, mask, color= viz_mask.cmap[c])
            imageio.imsave(os.path.join(out_path,'{}.png'.format(filename)), np.uint8(img))

    
def test_and_output(model,test_dataloader, out_path):
    labels = []
    for data in test_dataloader:
        data = data.to(device)
        output = model(data)
        pred = output.max(1, keepdim=False)[1].cpu().numpy()
        labels.append(pred)
    output_to_png(test_dataloader.dataset.filenames,
                    labels,
                    test_dataloader.dataset,
                    out_path
                    )

if __name__=='__main__':
    # args = arg_parse()
    # img_path = args.img_path
    # out_path = args.out_path
    img_path = './'
    out_path = './'
    # model = torch.load('hw1_p1.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = PRED(img_path)
    test_dataloader = DataLoader(test_dataset)
    model = torch.load('./5_73.pth')
    test_and_output(model, test_dataloader, out_path)
    
    
