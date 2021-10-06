import torch.optim as optim # Import optim 
import torchvision # Import torchvision 
import h5py #import h5py
from torch.utils.data import dataset # import dataset
from ESC10Customdataset_Train_2SB1 import LAEData_Train # Call Customdataloader to read Training Data
from ESC10Customdataset_Test_2SB1 import LAEData_Test # Call Customdataloader to read Test Data
from torch.utils.data import DataLoader # Import Dataloader 
import torchvision.transforms as transforms # Import Transform 
import pandas as pd # Import Pnadas 
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
#from Attention import SA_Module # Import Self attention module
from torch import optim, cuda # import optimizer  and CUDA
import random # import random 
import torch.nn.functional as F # Import nn.functional 
import time # import time 
import sys # Import System 
import os # Import OS
from pytorchtools import EarlyStopping
from torchvision import models
import warnings
SEED = 1234 # Initialize seed 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda') # Define device type 
num_classes=10 # Define Number of classes 
in_channel=1   # Define Number of Input Channels 
learning_rate=5e-5 # Define Learning rate 
batch_size=4 # Define Batch Size 
EPOCHS =1000   # Define maximum Number of Epochs
FC_Size=512
SFC_Size=512
Temp=3
alpha=0.7
N_models=6
lam=0.90
warnings.filterwarnings("ignore")
train_transformations = transforms.Compose([ # Training Transform 
    #transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
    ])
test_transformations = transforms.Compose([ # Test Transform 
    #transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
    ])
train_dataset=LAEData_Train(transform=train_transformations) # Create tensor of training data 
Test_Dataset=LAEData_Test(transform=test_transformations)# Create tensor of test dataset 
train_size = int(0.7 * len(train_dataset)) # Compute size of training data using (70% As Training and 30% As Validation)
valid_size = len(train_dataset) - train_size # Compute size of validation data using (70% As Training and 30% As Validation)
Train_Dataset,Valid_Dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size]) # Training and Validation Data After (70%-30%)Data Split 
#train_set,test_set=torch.utils.data.random_split(dataset,[6000,2639])
#Labels=pd.read_csv("Devlopment.csv")
train_loader=DataLoader(dataset=Train_Dataset,batch_size=batch_size,shuffle=True) # Create Training Dataloader 
valid_loader=DataLoader(dataset=Valid_Dataset,batch_size=batch_size,shuffle=True)# Create Validation Dataloader 
test_loader=DataLoader(dataset=Test_Dataset,batch_size=batch_size,shuffle=False) # Create Test Dataloader 
class Teacher(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(Teacher, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.features=Pre_Trained_Layers
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(2048,num_classes)  # Set output layer as an one output
    def forward(self,image):
        x1 = self.features(image)
        x4=self.avgpool(x1)
        x4=x4.view(x4.size(0),-1)
        x5=self.fc(x4)
        return x5
Teacher_Model=Teacher()
Teacher_Model=Teacher_Model.to(device)
Teacher_MODEL_SAVE_PATH = os.path.join("/home/mani/Desktop/AK/TASLP 2021/FB Model Training/TASLP_FInal_KD", 'Teacher_ESC10_CNN.pt') # Define Path to save the model 
Teacher_Model.load_state_dict(torch.load(Teacher_MODEL_SAVE_PATH))
class SB21(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(SB21, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.features=Pre_Trained_Layers
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(2048,num_classes)  # Set output layer as an one output
    def forward(self,image):
        x1 = self.features(image)
        x4=self.avgpool(x1)
        x4=x4.view(x4.size(0),-1)
        x5=self.fc(x4)
        return x5
SB21_Model=SB21()
SB21_Model=SB21_Model.to(device)
SB21_MODEL_SAVE_PATH = os.path.join("/home/mani/Desktop/AK/TASLP 2021/FB Model Training/TASLP_FInal_KD", 'S1_Teacher_ESC10_CNN.pt') # Define Path to save the model 
SB21_Model.load_state_dict(torch.load(SB21_MODEL_SAVE_PATH))
class SB22(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(SB22, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.features=Pre_Trained_Layers
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(2048,num_classes)  # Set output layer as an one output
    def forward(self,image):
        x1 = self.features(image)
        x4=self.avgpool(x1)
        x4=x4.view(x4.size(0),-1)
        x5=self.fc(x4)
        return x5
SB22_Model=SB22()
SB22_Model=SB22_Model.to(device)
SB22_MODEL_SAVE_PATH = os.path.join("/home/mani/Desktop/AK/TASLP 2021/FB Model Training/TASLP_FInal_KD", 'S2_Teacher_ESC10_CNN.pt') # Define Path to save the model 
SB22_Model.load_state_dict(torch.load(SB22_MODEL_SAVE_PATH))
class Student(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(Student, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.features=Pre_Trained_Layers
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(512,num_classes)  # Set output layer as an one output
    def forward(self,image):
        x1 = self.features(image)
        x4=self.avgpool(x1)
        x4=x4.view(x4.size(0),-1)
        x5=self.fc(x4)
        return x5
Student_Model=Student()
Student_Model=Student_Model.to(device)
Student_optimizer = optim.Adam(Student_Model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function 
norm_type=1
def normalization(x,norm_dims,epsilon=1e-5):
	mean = x.mean(dim=norm_dims, keepdim=True)
	var = x.var(dim=norm_dims, keepdim=True)
	std = (var + epsilon).sqrt()
	# normalize the features, i.e., remove the moments
	x = (x - mean) / std
	return x, mean, std
def moex(x,SB21_Data,SB22_Data, y):
    norm_dims=[1,2,3]
    x, mean, std = normalization(x,norm_dims)
    SB21_Data,SB21_Mean, SB21_Std=normalization(SB21_Data,norm_dims)
    SB22_Data,SB22_Mean, SB22_Std=normalization(SB22_Data,norm_dims)
    ex_index = torch.randperm(x.shape[0])
    x = x*std[ex_index] + mean[ex_index]
    SB21_Data=SB21_Data*SB21_Std[ex_index]+SB21_Mean[ex_index]
    SB22_Data=SB22_Data*SB22_Std[ex_index]+SB22_Mean[ex_index]
    y_b = y[ex_index]
    return x,SB21_Data,SB22_Data, y, y_b
def interpolate_loss(output, y, y_b, loss_func, lam):
	return lam*loss_func(output, y) + (1. - lam)*loss_func(output, y_b)
def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc
def WEIEN_Loss(Predicted_FB_Label,Predicted_SB1_Label,Predicted_SB2_Label,Predicted_Student_Label,N_models,y,device,criterion,y_b):
        FB_Prob=F.softmax(Predicted_FB_Label,dim=1)
        SB1_Prob=F.softmax(Predicted_SB1_Label,dim=1)
        SB2_Prob=F.softmax(Predicted_SB2_Label,dim=1)
        FB_Loss=criterion(Predicted_FB_Label,y)
        #FB_Loss=interpolate_loss(Predicted_FB_Label,y,y_b,criterion,lam)
        SB1_Loss=criterion(Predicted_SB1_Label,y)
        #SB1_Loss=interpolate_loss(Predicted_SB1_Label,y,y_b,criterion,lam)
        SB2_Loss=criterion(Predicted_SB2_Label,y)
        #SB2_Loss=interpolate_loss(Predicted_SB2_Label,y,y_b,criterion,lam)

        W1=torch.exp(FB_Loss)
        W2=torch.exp(SB1_Loss)
        W3=torch.exp(SB2_Loss)
        FW1=1-((W1)/(W1+W2+W3))
        FW2=1-((W2)/(W1+W2+W3))
        FW3=1-((W3)/(W1+W2+W3))
        AVG_Prob=(FW1*FB_Prob+FW2*SB1_Prob+FW3*SB2_Prob)
        Student_Prob=F.softmax(Predicted_Student_Label,dim=1)
        Student_Loss=F.kl_div(Student_Prob,AVG_Prob)
        #print(Total_E_Loss)
        return Student_Loss    
def AVGEN_Loss(Predicted_FB_Label,Predicted_SB1_Label, Predicted_SB2_Label,Predicted_Student_Label,N_models,device):
        FB_Prob=F.softmax(Predicted_FB_Label,dim=1)
        SB1_Prob=F.softmax(Predicted_SB1_Label,dim=1)
        SB2_Prob=F.softmax(Predicted_SB2_Label,dim=1)
        Student_Prob=F.softmax(Predicted_Student_Label,dim=1)
        AVG_Prob=(FB_Prob+SB1_Prob+SB2_Prob)/N_models
        #AVG_Prob=F.softmax(AVG_Prob,dim=1)
        Student_Loss=F.kl_div(Student_Prob,AVG_Prob)
        #print(Total_E_Loss)
        return Student_Loss
def MINEN_Loss(Predicted_FB_Label,Predicted_SB1_Label,Predicted_SB2_Label,Predicted_Student_Label,N_models,y,device,criterion):
        FB_Prob=F.softmax(Predicted_FB_Label,dim=1)
        SB1_Prob=F.softmax(Predicted_SB1_Label,dim=1)
        SB2_Prob=F.softmax(Predicted_SB2_Label,dim=1)
        FB_Loss=criterion(Predicted_FB_Label,y) 
        SB1_Loss=criterion(Predicted_SB1_Label,y)
        SB2_Loss=criterion(Predicted_SB2_Label,y)
        Student_Prob=F.softmax(Predicted_Student_Label,dim=1)
        L_Array=torch.tensor([FB_Loss,SB1_Loss,SB2_Loss])
        L_Array=L_Array.numpy()
        Min_Loss_Index=np.argmin(L_Array)
        if Min_Loss_Index==0:
            Student_Loss=F.kl_div(Student_Prob,FB_Prob)
        if Min_Loss_Index==1:
            Student_Loss=F.kl_div(Student_Prob,SB1_Prob)
        if Min_Loss_Index==2:
            Student_Loss=F.kl_div(Student_Prob,SB2_Prob)
        return Student_Loss            
def train(Teacher_Model, Student_Model,SB21_Model, SB22_Model,device,iterator, optimizer, criterion,N_models):  
    early_stopping = EarlyStopping(patience=7, verbose=True)
    epoch_loss = 0
    epoch_acc = 0
    count=0
    Student_Model.train() # call model object for training 
    for (x1,SB21_Data,SB22_Data,y) in iterator:
        x,SB21_Data,SB22_Data,y,y_b=moex(x1,SB21_Data,SB22_Data, y)
        #print(x.var([2,3],keepdim=True))
        x=x.float()
        #print(x.type())
        x=x.to(device)
        y=y.to(device)
        y_b=y_b.to(device)
        Predicted_Student_Label=Student_Model(x)
        Predicted_Student_Prob=F.softmax(Predicted_Student_Label,dim=1)
        Predicted_Teacher_Label=Teacher_Model(x)
        Predicted_Teacher_Prob=F.softmax(Predicted_Teacher_Label,dim=1)
        Predicted_SB21_Label=SB21_Model(x)
        Predicted_SB21_Prob=F.softmax(Predicted_SB21_Label,dim=1)
        Predicted_SB22_Label=SB22_Model(x)
        Predicted_SB22_Prob=F.softmax(Predicted_SB22_Label,dim=1)
        x=x.detach().cpu()
        SB21_Data=SB21_Data.detach().cpu()
        SB22_Data=SB22_Data.detach().cpu()
        IP_Loss=interpolate_loss(Predicted_Student_Prob,y,y_b,criterion,lam)
        optimizer.zero_grad() # Initialize gredients as zeros 
        #KL_Loss=F.kl_div(Predicted_Teacher_Prob, Predicted_Student_Prob)
        #KL_Loss=AVGEN_Loss(Predicted_Teacher_Label,Predicted_SB21_Label, Predicted_SB22_Label,Predicted_Student_Label,N_models,device)
        #KL_Loss=MINEN_Loss(Predicted_Teacher_Label,Predicted_SB21_Label,Predicted_SB22_Label,Predicted_Student_Label,N_models,y,device,criterion)
        KL_Loss=WEIEN_Loss(Predicted_Teacher_Label,Predicted_SB21_Label,Predicted_SB22_Label,Predicted_Student_Label,N_models,y,device,criterion,y_b)
        #print(x.shape)
        x1=x1.float()
        x1=x1.to(device)
        Predicted_Train_Label=Student_Model(x1)
        Predicted_Train_Label=F.softmax(Predicted_Train_Label,dim=1)
        loss = IP_Loss+KL_Loss
        acc = calculate_accuracy(Predicted_Train_Label,y) # training accuracy 
        #print("Training Iteration Number=",count)
        loss.backward() # backpropogation 
        optimizer.step() # optimize the model weights using an optimizer 
        epoch_loss += loss.item() # sum of training loss
        epoch_acc += acc.item() # sum of training accuracy  
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(Teacher_Model, Student_Model,device,iterator, criterion): # Evaluate Validation accuracy 
    #print("Validation Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    Student_Model.eval() # call model object for evaluation 
    with torch.no_grad(): # Without computation of gredient 
        for (x1,SB21_Data,SB22_Data,y) in iterator:
            x,SB21_Data,SB22_Data,y,y_b=moex(x1,SB21_Data,SB22_Data,y)
            #print(x.var([2,3],keepdim=True))
            x=x.float()
            #print(x.type())
            x=x.to(device)
            y=y.to(device)
            y_b=y_b.to(device)
            Predicted_Student_Prob=Student_Model(x)
            Predicted_Student_Prob=F.softmax(Predicted_Student_Prob,dim=1)
            Predicted_Teacher_Prob=Teacher_Model(x)
            Predicted_Teacher_Prob=F.softmax(Predicted_Teacher_Prob,dim=1)
            x=x.detach().cpu()
            IP_Loss=interpolate_loss(Predicted_Student_Prob,y,y_b,criterion,lam)
            #optimizer.zero_grad() # Initialize gredients as zeros 
            KL_Loss=F.kl_div(Predicted_Teacher_Prob, Predicted_Student_Prob)
            #print(x.shape)
            x1=x1.float()
            x1=x1.to(device)
            Predicted_Train_Label=Student_Model(x1)
            Predicted_Train_Label=F.softmax(Predicted_Train_Label,dim=1)
            CEL_Loss=criterion(Predicted_Train_Label,y)
            #loss = IP_Loss+KL_Loss
            loss=CEL_Loss
            acc = calculate_accuracy(Predicted_Train_Label,y) # training accuracy 
            #print("Training Iteration Number=",count)
            #loss.backward() # backpropogation 
            #optimizer.step() # optimize the model weights using an optimizer 
            epoch_loss += loss.item() # sum of training loss
            epoch_acc += acc.item() # sum of training accuracy  
    return epoch_loss / len(iterator), epoch_acc / len(iterator) 
MODEL_SAVE_PATH = os.path.join("/home/mani/Desktop/AK/TASLP 2021/FB Model Training/TASLP_FInal_KD", '2SB_Ensemble_Student_ESC10_CNN.pt') # Define Path to save the model 
Temp=np.zeros([EPOCHS,6]) # Temp Matrix to Store all model accuracy, loss and time parameters 
print("Student ESC10 Model is in Training Mode") 
print("---------------------------------------------------------------------------------------------------------------------")   
early_stopping = EarlyStopping(patience=7, verbose=True) # early Stopping Criteria
for epoch in range(EPOCHS):
    start_time=time.time() # Compute Start Time 
    train_loss, train_acc = train(Teacher_Model, Student_Model,SB21_Model, SB22_Model,device,train_loader,Student_optimizer, criterion,N_models) # Call Training Process 
    train_loss=round(train_loss,2) # Round training loss 
    train_acc=round(train_acc,2) # Round training accuracy 
    valid_loss, valid_acc = evaluate(Teacher_Model,Student_Model,device,valid_loader, criterion) # Call Validation Process 
    valid_loss=round(valid_loss,2) # Round validation loss
    valid_acc=round(valid_acc,2) # Round accuracy 
    end_time=(time.time()-start_time) # Compute End time 
    end_time=round(end_time,2)  # Round End Time 
    print(" | Epoch=",epoch," | Training Accuracy=",train_acc*100," | Validation Accuracy=",valid_acc*100," | Training Loss=",train_loss," | Validation_Loss=",valid_loss,"Time Taken(Seconds)=",end_time,"|")
    print("---------------------------------------------------------------------------------------------------------------------")
    Temp[epoch,0]=epoch # Store Epoch Number 
    Temp[epoch,1]=train_acc # Store Training Accuracy 
    Temp[epoch,2]=valid_acc # Store Validation Accuracy 
    Temp[epoch,3]=train_loss # Store Training Loss 
    Temp[epoch,4]=valid_loss # Store Validation Loss 
    Temp[epoch,5]=end_time # Store Running Time of One Epoch 
    early_stopping(valid_loss,Student_Model) # call Early Stopping to Prevent Overfitting 
    if early_stopping.early_stop:
        print("Early stopping")
        break
    Student_Model.load_state_dict(torch.load(MODEL_SAVE_PATH))
np.save('Student_ESC10_CNN_Parameters',Temp) # Save Temp Array as numpy array 
Student_Model.load_state_dict(torch.load(MODEL_SAVE_PATH)) # load the trained model 
test_loss, test_acc = evaluate(Teacher_Model, Student_Model,device,test_loader, criterion) # Compute Test Accuracy on Unseen Signals 
#test_loss=round(test_loss,2)# Round test loss
#test_acc=round(test_acc,2) # Round test accuracy 
print("|Test Loss=",test_loss,"Test Accuracy=",test_acc*100) # print test accuracy     

