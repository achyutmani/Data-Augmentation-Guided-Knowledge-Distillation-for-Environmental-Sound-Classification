# Acoustic Event Detection Using Knowledge Distillation from Attention-Based Subband Specilist Deep Model 
import torch.optim as optim
import torchvision
import h5py
from torch.utils.data import dataset
from ESC10Customdataset_Train_FB import LAEData_Train
from ESC10Customdataset_Test_FB import LAEData_Test
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import torch # Import Torch 
import torch.nn as nn # Import NN module from Torch 
from torchvision.transforms import transforms# Import transform module from torchvision 
from torch.utils.data import DataLoader # Import dataloader from torch 
from torch.optim import Adam # import optimizer module from torch 
from torch.autograd import Variable # Import autograd from torch 
import numpy as np # Import numpy module 
import torchvision.datasets as datasets #Import dataset from torch 
from Attention import PAM_Module # import position attention module 
from Attention import CAM_Module # import channel attention module
from torch import optim, cuda # import optimizer
import random
import torch.nn.functional as F
import random
import time
import sys
import os
from pytorchtools import EarlyStopping
from sklearn.metrics import confusion_matrix
from torchvision import models
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda')
num_classes=10
in_channel=1
num_class=10
learning_rate=5e-5
batch_size=8
EPOCHS =100
train_transformations = transforms.Compose([ # Training Transform 
    #transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])
test_transformations = transforms.Compose([ # Test Transform 
    #transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])
train_dataset=LAEData_Train(transform=train_transformations)
Test_Dataset=LAEData_Test(transform=test_transformations)
train_size = int(0.7 * len(train_dataset))
valid_size = len(train_dataset) - train_size
Train_Dataset,Valid_Dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
#train_set,test_set=torch.utils.data.random_split(dataset,[6000,2639])
#Labels=pd.read_csv("Devlopment.csv")
train_loader=DataLoader(dataset=Train_Dataset,batch_size=batch_size,shuffle=True)
valid_loader=DataLoader(dataset=Valid_Dataset,batch_size=batch_size,shuffle=False)
test_loader=DataLoader(dataset=Test_Dataset,batch_size=batch_size,shuffle=False)
class Teacher_Model(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(Teacher_Model, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        #Pre_Trained_Layers = list(models.resnet34(pretrained=True).children())[:-4]
        #Pre_Trained_Layers = models.resnet34(pretrained=True) # Initialize model layers and weights
        #print(Pre_Trained_Layers)
        self.features=Pre_Trained_Layers
        #self.CAM=CAM_Module(512)
        #self.CAM=CAM_Module(512)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))

        #self.features.Flat=nn.Flatten()
        self.fc=nn.Linear(2048,num_classes)  # Set output layer as an one output

    def forward(self,image):
        x1 = self.features(image)
        #x1=self.CAM(x)
        #x2=self.CAM(x1)
        #x3=x1+x2
        x4=self.avgpool(x1)
        #x4=x3.view(x3.shape[0],-1)
        x4=x4.view(x4.size(0),-1)
        #x4=torch.flatten(x4)
        #print(x4.shape)
        #x4=torch.unsqueeze(x4,-1)
        #print(x4.shape)
        x5=self.fc(x4)
        return x5
Teacher_Model=Teacher_Model()        
Teacher_Model=Teacher_Model.to(device)
Teacher_optimizer = optim.Adam(Teacher_Model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function 
def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    #T=3
    #LL=nn.KLDivLoss()((F.log_softmax(fx/T,dim=1)),(F.softmax(fx/T,dim=1)))
    #print(fx.shape)
    return acc
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc
def evaluate(model,device,iterator, criterion): # Evaluate Validation accuracy 
    #print("Validation Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    model.eval() # call model object for evaluation 
    
    with torch.no_grad(): # Without computation of gredient 
        for (x, y) in iterator:
            x=x.float()
            x=x.to(device) # Transfer data to device 
            y=y.to(device) # Transfer label  to device 
            count=count+1
            Predicted_Label = model(x) # Predict claa label 
            loss = criterion(Predicted_Label, y) # Compute Loss 
            acc = calculate_accuracy(Predicted_Label, y) # compute Accuracy 
            #print("Validation Iteration Number=",count)
            epoch_loss += loss.item() # Compute Sum of  Loss 
            epoch_acc += acc.item() # Compute  Sum of Accuracy 
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator) 

SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join("/home/mani/Desktop/AK/TASLP 2021/FB Model Training/TASLP_FInal_KD", 'S2_Teacher_ESC10_CNN.pt') # Define Path to save the model 
Teacher_Model.load_state_dict(torch.load(MODEL_SAVE_PATH))
# test_loss, test_acc = evaluate(Teacher_Model, device, test_loader, criterion)
# test_loss=round(test_loss,2)
# test_acc=round(test_acc,2)
# print("|Test Loss=",test_loss,"Test Accuracy=",test_acc*100)
def get_all_preds(model,loader):
     all_preds = torch.tensor([])
     all_preds=all_preds.to(device)
     all_actual=torch.tensor([])
     all_actual=all_actual.to(device)
     for batch in loader:
         images, labels = batch
         images=images.to(device)
         labels=labels.to(device)
         labels=labels.float()  
         preds = (nn.functional.softmax(model(images),dim=1)).max(1,keepdim=True)[1]
         #fx.max(1, keepdim=True)[1]
#         #print(preds)
#         #print(labels)
#         #dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
         all_preds = torch.cat((all_preds, preds.float()),dim=0)
#         #print(all_preds)
         all_actual = torch.cat((all_actual,labels),dim=0)
     return all_preds,all_actual
with torch.no_grad():
#train_preds,train_actual = get_all_preds(Teacher_Model,train_loader)
    test_preds,test_actual = get_all_preds(Teacher_Model,test_loader)
# Train_CM = confusion_matrix(train_actual.cpu().numpy(),train_preds.cpu().numpy())
# Test_CM = confusion_matrix(test_actual.cpu().numpy(),test_preds.cpu().numpy())
# #print(Train_CM)
pytorch_total_params = sum(p.numel() for p in Teacher_Model.parameters() if p.requires_grad)
print(pytorch_total_params/1000000)
# #print(Test_CM)
# import itertools
# import numpy as np
# import matplotlib.pyplot as plt
# classes= ["Baby Cry","Chainsaw","Clock Tick","Dog Bark","Fire Cracking","Helicopter","Person Sneeze","Rain","Rooster","Sea Waves"]
# def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=90,size=12)
#     plt.yticks(tick_marks, classes,rotation=45,size=12)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label',size=12)
#     plt.xlabel('Predicted label',size=12)
# plt.figure(figsize=(7,7))
# plt.rcParams['font.size'] = 12
# plot_confusion_matrix(Test_CM,classes) 
# plt.tight_layout()
# plt.savefig('Dual_CM.png')
# plt.show()  
#print(Teacher_Model.fc) 
#======================================================================Implement GRAD-CAM=================================================================================
feature_extractor = torch.nn.Sequential(*list(Teacher_Model.children())[:-1])
#print(feature_extractor)
# def feature_Map(model,Input_Data):
# 	class SaveOutput:
# 	    def __init__(self):
# 	        self.outputs = []
# 	    def __call__(self, module, module_in, module_out):
# 	        self.outputs.append(module_out) 
# 	    def clear(self):
# 	        self.outputs = [] 
# 	save_output= SaveOutput()
# 	hook_handles=[]
# 	handle1 = model.fc.register_forward_hook(save_output)
# 	hook_handles.append(handle1)
# 	Temp= model(Input_Data)
# 	def module_output_to_numpy(tensor):
# 		return tensor.detach().to('cpu').numpy()
# 	F_Map = module_output_to_numpy(save_output.outputs[0])
# 	F_Map=torch.from_numpy(F_Map).to(device)	
# 	return F_Map
Feature=np.zeros((batch_size,2048))
flag=0
for batch in test_loader:
    Data,Label=batch
    Data=Data.to(device)
    FM1=feature_extractor(Data)
    FM1=torch.squeeze(FM1)
    if flag==0:
        Feature=FM1.detach().cpu().numpy()
    else:
        Feature=np.vstack((Feature,FM1.detach().cpu().numpy()))    
    flag=flag+1
    #FM1=FM1.view(FM1.size(0),-1)
    #FM1=torch.flatten(FM1)
    
    #print(FM1.shape)	
#print(Feature.shape)
#print(Feature)   
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(Feature) 
target_ids = range(10)
test_preds=(test_preds.detach().cpu().numpy())
#print(test_preds)
from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'maroon', 'orange', 'purple'
classes= ["Baby Cry","Chainsaw","Clock Tick","Dog Bark","Fire Cracking","Helicopter","Person Sneeze","Rain","Rooster","Sea Waves"]
for i, c, label in zip(target_ids,colors,classes):
    plt.scatter(X_2d[np.where(test_preds == i), 0], X_2d[np.where(test_preds == i), 1], c=c, label=label,marker='o',s = 200)
plt.legend()
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel('Dimension 1',size=30)
plt.ylabel('Dimension 2',size=30)
plt.title('t-SNE Plot for ESC-10 Dataset')
figure = plt.gcf()  # get current figure
figure.set_size_inches(22,10) # set figure's size manually to your full screen (32x18)
plt.savefig('/home/mani/Desktop/AK/TASLP 2021/FB Model Training/TASLP_FInal_KD/t_SNE Teacher/S2_Teacher_tSNE_ESC10.png', bbox_inches='tight') # bbox_inches removes extra white spaces

