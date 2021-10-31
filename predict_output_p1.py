import argparse
from PIL import Image
import torch
import glob
import os
from torch.utils.data import DataLoader,Dataset
import pandas as pd


class PRED(Dataset):
    def __init__(self, root, transform=None):
        self.X = None
        self.filenames = []
        self.filepaths =  glob.glob(root+'/*.png')
        self.root = root
        self.transform = transform

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
        help='path to image'
    )
    args = parser.parse_args()
    return args

def output_to_csv(filenames, labels):
    pred = {"image_id":[],"label":[]}
    df = pd.DataFrame(pred)
    df["image_id"] = filenames
    df["label"] = labels    
    print(df)
    df.to_csv('pred.csv', index = False)
    
def test_and_output(model,test_dataloader):
    labels = []
    for data in test_dataloader:
        data = data.to(device)
        output = model(data)
        pred = output.max(1, keepdim=False)[1].cpu().numpy()
        labels.append(pred)
    output_to_csv(test_dataloader.dataset.filenames,labels)

if __name__=='__main__':
    # args = arg_parse()
    # img_path = args.img_path
    # model = torch.load('hw1_p1.pth')
    # img = Image.open(img_path)
    # # out = model(img)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = PRED('HW1/p1_data/v_6',transform=(
        
    ))
    test_dataloader = DataLoader(test_dataset)
    model = torch.load('HW1/p1GOOD/5_73.pth')
    test_and_output(model, test_dataloader)
    
    
