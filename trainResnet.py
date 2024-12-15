from model import Resnet
import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils import data
import logging

logger = logging.getLogger('training_logger')
logger.setLevel(logging.INFO) 

class Accumulator:
    def __init__(self, n):
        self.count = 0
        self.data = [0.0] * n

    def add(self,*args):
        self.count += 1
        self.data = [a + float(b) for a, b in zip(self.data,args)]

    def reset(self):
        self.count = 0 
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

def f1score(pred,label):

    with torch.no_grad():
        pred = torch.argmax(pred,dim=-1)
        TP = sum((label == 1) & (pred == 1))#真正
        FP = sum((label == 0) & (pred == 1))#假正
        FN = sum((label == 1) & (pred == 0))#假反
        precision = TP/(TP+FP+1e-9)
        recall = TP/(TP+FN+1e-9)
        return 2 * precision * recall/(precision + recall)


    


def trainresnet(net, num_epochs, optimizer, scheduler, device, train_iter, valid_iter):
    
    print('training on', device)
    net.to(device)
    loss = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        
        net.train()
        lr = optimizer.param_groups[0]['lr']
        print(f'The learning rate is: {lr}')

        for i, (X, y) in enumerate(train_iter):
            torch.cuda.empty_cache()

            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
        
            l = loss(y_hat, y)
            
            l.backward()
            optimizer.step()
            print(f'iter {i}: loss:{l.cpu()}')
            logger.info(f'iter {i+1}: loss:{l.cpu()}')
        with torch.no_grad():
            testloss,f1scr = evaluate(net, valid_iter,device)
        print(f'epoch {epoch}: test entropy:{testloss},f1-score:{f1scr}')
        logger.info(f'Epoch {epoch + 1} test loss: {testloss}')
        # checkpoint = {
		#     'epoch': epoch,
		#     'model_state_dict': net.state_dict(),
		#     'optimizer_state_dict': optimizer.state_dict(),
		#     # 'loss': loss.item()
	    # }
        # torch.save(checkpoint, 'checkpoint.pth')
        scheduler.step()
    torch.save(net.state_dict(), 'resnet_state_dict.pth')

# net = Resnet()
net = Resnet()

X_train = torch.load('X_train_all.pt')
y_train = torch.load('y_train_all.pt')
X_valid = torch.load('X_valid.pt')
y_valid = torch.load('y_valid.pt')

train_iter = DataLoader(data.TensorDataset(X_train,y_train),batch_size=40,shuffle=True)
valid_iter = DataLoader(data.TensorDataset(X_valid,y_valid),batch_size=20,shuffle=False)

loss = nn.MSELoss()
# y_train, X_test, y_tes


device = torch.device(f'cuda:0')
print(device)
torch.cuda.empty_cache()
optim = torch.optim.Adam(net.parameters(),lr=0.00001)
# scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=128)
scheduler = ExponentialLR(optim, gamma=0.9)
trainresnet(net,num_epochs=120,optimizer=optim,scheduler=scheduler,device=device,train_iter=train_iter,valid_iter=valid_iter)

