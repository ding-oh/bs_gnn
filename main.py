import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from generateBindingsiteGraph import *

# define path location
proj_p = '/home/dongwoo/project/geo/BS_prediction'
data_p = '/home/dongwoo/project/geo/scPDB_m/clean_set'

validCnt, invalidCnt, numTrainSet = 0, 0, 9
data_list = []

for i in range(numTrainSet): # using set0 for test
    print(f'[INFO ] set {i}')
    for seqInfo in tqdm(open(f'{proj_p}/valid_data.set/test_seq.info', 'r').readlines()[1:]):
        try:
            result = convertBS2Graph(seqInfo=seqInfo,
                            file_p=f'{data_p}/set_{0}/{seqInfo.split(",")[0]}',
                            pdb=seqInfo.split(',')[0],
                            chain=seqInfo.split(',')[1]
                            )
            node_attr, edge_idx, edge_weight = result
            y = [ int(i) for i in seqInfo.split(',')[-1][:-1] ]
            y = torch.tensor(y, dtype=torch.long)
            dtmp = Data(x=node_attr, edge_index=edge_idx, edge_weight=edge_weight, y=y)
            print('\n', dtmp)
            data_list.append(dtmp)
            # print("data_preparation :", seqInfo.split(',')[0], seqInfo.split(',')[1], "...ok")
        except Exception as E:
            print(E)


train_set, val_set, test_set = data_list[:1250], data_list[1250:1430], data_list[1430:] # data_length = 1791
print(f"Number of training set: {len(train_set)}")
print(f"Number of Validaation set: {len(val_set)}")
print(f"Number of test set: {len(test_set)}")

train_loader = DataLoader(train_set, batch_size=100, shuffle=True, drop_last=False)
test_loader = DataLoader(test_set, batch_size=100, shuffle=True, drop_last=False)
val_loader = DataLoader(val_set, batch_size=100, shuffle=False, drop_last=False)

for batch in train_loader:
    print(batch)
    print(batch.batch)
    print(batch.num_graphs)
    print(batch.num_node_features)
    break

data = data_list[0]
print("data.keys : ", data.keys)
print("data.num_node_features", data.num_node_features)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)
model = myGCN(in_channel=34, hidden_layer_size=70)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.CrossEntropyLoss()

def test(loader):
    model.eval()
    error = 0.0
    out_all = []
    true = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.to(device))
        out = out.argmax(dim=1)
        tmp = (out - data.y) ** 2
        error += tmp.sum().item()  # Check against ground-truth labels.

        out_all.extend([x.item() for x in out])
        true.extend([x.item() for x in data.y])
    return error / len(loader.dataset), out_all, true  # Derive ratio of correct predictions.

def train():
    model.train()
    for idx, batch in enumerate(train_loader):
        out = model(batch.to(device))
        #out = out.argmax(dim=1)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        if idx%100 == 0:
            print(f"IDX: {idx:5d}\tLoss: {loss:.4f}")

train_acc_list = []
val_acc_list = []
test_acc_list = []

for epoch in range(1, 50):
    print("=" * 100)
    print("Epoch: ", epoch)

    train()

    train_acc, out_tmp, true_tmp = test(train_loader)
    train_acc_list.append(train_acc)

    test_acc, out_all, true_all = test(test_loader)
    test_acc_list.append(test_acc)

    val_acc, val_pred, val_true = test(val_loader)
    val_acc_list.append(val_acc)

    print("-" * 100)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

torch.save(model, '/home/dongwoo/project/geo/BS_prediction/model.pt')

model = torch.load('/home/dongwoo/project/geo/BS_prediction/model.pt')

experimental = [x for x in true_all]#out_all = [x.detach().numpy() for x in out_all]
prediction = [x for x in out_all]

with open("binary_predicted.txt", "w") as f:
    f.writelines([x for x in out_all])

import matplotlib.pyplot as plt
def draw_loss_change(train_loss, val_loss, test_loss):
  plt.figure(figsize=(8,8)) # 빈 그림을 정의
  plt.plot(train_loss, color = 'r', label = 'Train loss') # training loss
  plt.plot(val_loss, color = 'b', label = 'Val loss') # validation set loss
  plt.plot(test_loss, color = 'g',  label = 'Test loss') # test set loss
  plt.xlabel("Epoch")#out_all = [x.detach().numpy() for x in out_all]
  plt.ylabel("Loss")
  plt.legend(loc='best') # label을 표시 하겠다.

draw_loss_change(train_acc_list, val_acc_list, test_acc_list)


experimental = [x for x in true_all]
prediction = [x for x in out_all]

plt.figure(figsize=(8, 8))
plt.scatter(experimental, prediction, marker = '.')
plt.plot(range(-3, 4), range(-3, 4), 'r--')
plt.xlabel("experimental value(logBB)", fontsize='xx-large')
plt.ylabel("Prediction (logBB)", fontsize='xx-large')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
