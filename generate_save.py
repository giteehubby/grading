import torch
import pickle
from get_data import get_data

train_path = 'train'
X_train, y_train,X_valid, y_valid  = get_data(train_path,augrate=2,valid_num=60)

print(f'X_train.shape:{X_train.shape}')
print(f'y_train.shape:{y_train.shape}')
print(f'X_valid.shape:{X_valid.shape}')
print(f'y_valid.shape:{y_valid.shape}')
torch.save(X_train,'X_train_all.pt')
torch.save(y_train,'y_train_all.pt')
torch.save(X_valid,'X_valid.pt')
torch.save(y_valid,'y_valid.pt')
print('successfully saved!')

# path = 'test'
# X_test,filenms = get_data(path,is_train=False)

# print(f'X_test.shape:{X_test.shape}')

# torch.save(X_test,'X_test.pt')
# with open('filenms.pkl', 'wb') as f:
#     pickle.dump(filenms, f)