from torchvision import transforms
import torch
from PIL import Image
from model import Resnet
from vit import ViT
import pickle
from torch.utils.data import DataLoader
from torch.utils import data
import csv


def predict(net,test_X,device,filenms):
    net.to(device)
    print(f'predict on {device}')
    y_preds = []
    test_iter = DataLoader(data.TensorDataset(test_X,),batch_size=20,shuffle=False,drop_last=False)
    for (X,) in test_iter:

        X = X.to(device)
        pred = net(X)
        pred = torch.argmax(pred,dim=-1)
        y_preds.append(pred)
        print(pred.shape)
    y_pred = torch.cat(y_preds,dim=0)
    print(y_pred.shape)
    write_result(y_pred,filenms)

def write_result(y_pred,filenms):
    csv_file_name = "result.csv"
 
    # 打开（或创建）CSV文件，并写入数据
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image','Predict'])
        for i in range(len(y_pred)):
            writer.writerow([filenms[i],y_pred[i].item()])



# net = ViT(num_encoder=5,embedding_size=128,patch_size=16,img_size=512,num_channel=3,num_heads=8,dropout=0.1,
#           normshape=[1025,128],ffn_hidden_size=256,dec_hidden_size=64)

# net = ViT(num_encoder=2,embedding_size=512,patch_size=16,img_size=512,num_channel=3,num_heads=8,dropout=0.1,
#           normshape=[1025,512],ffn_hidden_size=512,dec_hidden_size=512)

net = Resnet()
device = torch.device('cuda')
state_dict = torch.load('resnet_state_dict.pth')
net.load_state_dict(state_dict)
X = torch.load('X_test.pt')
with open('filenms.pkl', 'rb') as f:
    filenms = pickle.load(f)
predict(net,X,device,filenms)