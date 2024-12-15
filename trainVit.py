from vit import ViT
import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
from trainResnet import f1score 



class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self,*args):
        self.data = [a + float(b) for a, b in zip(self.data,args)]

    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self,i):
        return self.data[i]

def evaluate(net,data_iter,device):
    if isinstance(net,nn.Module):
        net.eval()
    metric = Accumulator(2)

    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        net.to(device)
        pred = net(X)
        f1 = f1score(pred,y)
        metric.add(nn.CrossEntropyLoss()(pred,y).cpu(),f1.cpu())
    return metric[0],metric[1]/metric.count
    


def trainvitnet(net, num_epochs, optimizer, scheduler, device, train_iter, valid_iter):
    
    print('training on', device)
    torch.cuda.empty_cache()
    net.to(device)
    loss = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):

        
        net.train()
        lr = optimizer.param_groups[0]['lr']
        print(f'The learning rate is: {lr}')
        i = 0

        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device).to(torch.long)
            y_hat = net(X)
         
            l = loss(y_hat, y)
            
            l.backward()
            optimizer.step()
            i += 1
            print(f'iter {i}: loss:{l.cpu()}')
        with torch.no_grad():
            testloss,f1scr = evaluate(net, valid_iter,device)
        print(f'epoch {epoch}: test entropy:{testloss},f1-score:{f1scr}')
        # checkpoint = {
		#     'epoch': epoch,
		#     'model_state_dict': net.state_dict(),
		#     'optimizer_state_dict': optimizer.state_dict(),
		#     # 'loss': loss.item()
	    # }
        # torch.save(checkpoint, 'checkpoint.pth')
        scheduler.step()
    torch.save(net.state_dict(), 'vit_state_dict.pth')

# net = Resnet()
net = ViT(num_encoder=4,embedding_size=256,patch_size=16,img_size=512,num_channel=3,num_heads=8,dropout=0.1,
          normshape=[1025,256],ffn_hidden_size=256,dec_hidden_size=48)

X_train = torch.load('X_train_all.pt')
y_train = torch.load('y_train_all.pt')
X_valid = torch.load('X_valid.pt')
y_valid = torch.load('y_valid.pt')

train_iter = DataLoader(data.TensorDataset(X_train,y_train),batch_size=20,shuffle=True)
valid_iter = DataLoader(data.TensorDataset(X_valid,y_valid),batch_size=20,shuffle=False)


device = torch.device(f'cuda')
print(device)
torch.cuda.empty_cache()
optim = torch.optim.Adam(net.parameters(),lr=0.00001)
# scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=128)
scheduler = ExponentialLR(optim, gamma=0.95)
trainvitnet(net,num_epochs=100,optimizer=optim,scheduler=scheduler,device=device,train_iter=train_iter,valid_iter=valid_iter)

