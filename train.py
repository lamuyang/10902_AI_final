import os
os.environ['CUDA_ENABLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from dataset.dataset import SeedlingData
from torch.autograd import Variable
from models.coatnet import coatnet_0
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True 


modellr = 1e-4
BATCH_SIZE = 32
EPOCHS =  1000
# DEVICE = torch.device('cuda')
if torch.cuda.is_available():
	DEVICE = torch.device("cuda")
else:
	DEVICE = torch.device("cpu")




transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


dataset_train = SeedlingData('data/train', transforms=transform, train=True)
dataset_test = SeedlingData("data/train", transforms=transform_test, train=False)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)


criterion = nn.CrossEntropyLoss()

model_ft = coatnet_0()
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 14)
model_ft.to(DEVICE)


optimizer = optim.Adam(model_ft.parameters(), lr=modellr)
cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=20,eta_min=1e-9)




def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        # print(output)
        # print(target)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))



def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))
    return correct


# train
old_correct = 0
for epoch in range(1, EPOCHS + 1):
    N = 0
    train(model_ft, DEVICE, train_loader, optimizer, epoch)
    cosine_schedule.step()
    correct = val(model_ft, DEVICE, test_loader)
    if correct > old_correct or correct == old_correct:
        torch.save(model_ft,"model.pth")
        N == 0
    elif N == 3:
        break
    else:
        N+=1
    old_correct = correct 
    

