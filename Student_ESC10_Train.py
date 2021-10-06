import torch.optim as optim # Import optim 
import torchvision # Import torchvision 
import h5py #import h5py
from torch.utils.data import dataset # import dataset
from ESC10Customdataset_Train_FB import LAEData_Train # Call Customdataloader to read Training Data
from ESC10Customdataset_Test_FB import LAEData_Test # Call Customdataloader to read Test Data
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
from torch.optim.lr_scheduler import StepLR
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
learning_rate=2e-5 # Define Learning rate 
batch_size=8 # Define Batch Size 
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
        #Pre_Trained_Layers = list(models.resnet34(pretrained=True).children())[:-4]
        #Pre_Trained_Layers = models.resnet34(pretrained=True) # Initialize model layers and weights
        #print(Pre_Trained_Layers)
        self.features=Pre_Trained_Layers
        #self.PAM=PAM_Module(512)
        #self.CAM=CAM_Module(512)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))

        #self.features.Flat=nn.Flatten()
        self.fc=nn.Linear(2048,num_classes)  # Set output layer as an one output

    def forward(self,image):
        x1 = self.features(image)
        #x1=self.PAM(x1)
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
Teacher_Model=Teacher()
Teacher_Model=Teacher_Model.to(device)
Teacher_MODEL_SAVE_PATH = os.path.join("/home/mani/Desktop/AK/TASLP 2021/FB Model Training/TASLP_FInal_KD", 'Teacher_ESC10_CNN.pt') # Define Path to save the model 
Teacher_Model.load_state_dict(torch.load(Teacher_MODEL_SAVE_PATH))
class Student(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(Student, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        #Pre_Trained_Layers = list(models.resnet34(pretrained=True).children())[:-4]
        #Pre_Trained_Layers = models.resnet34(pretrained=True) # Initialize model layers and weights
        #print(Pre_Trained_Layers)
        self.features=Pre_Trained_Layers
        #self.PAM=PAM_Module(512)
        #self.CAM=CAM_Module(512)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))

        #self.features.Flat=nn.Flatten()
        self.fc=nn.Linear(512,num_classes)  # Set output layer as an one output

    def forward(self,image):
        x1 = self.features(image)
        #x1=self.PAM(x1)
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
Student_Model=Student()
Student_Model=Student_Model.to(device)
Student_optimizer = optim.Adam(Student_Model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function 
norm_type=1
def normalization(x,norm_dims,epsilon=1e-5):
# decide how to compute the moments
	#if norm_type == ’pono’:
		#norm_dims = [1]
	#elif norm_type == ’instance_norm’:
		#norm_dims = [2, 3]
	#else: # layer norm
		#norm_dims = [1, 2, 3]
	# compute the moments
	mean = x.mean(dim=norm_dims, keepdim=True)
	var = x.var(dim=norm_dims, keepdim=True)
	std = (var + epsilon).sqrt()
	# normalize the features, i.e., remove the moments
	x = (x - mean) / std
	return x, mean, std
def moex(x, y):
    norm_dims=[1,2,3]
    x, mean, std = normalization(x,norm_dims)
    #print(x)
    ex_index = torch.randperm(x.shape[0])
    x = x*std[ex_index] + mean[ex_index]
    y_b = y[ex_index]
    return x, y, y_b
def interpolate_loss(output, y, y_b, loss_func, lam):
	return lam*loss_func(output, y) + (1. - lam)*loss_func(output, y_b)
def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc
scheduler = StepLR(Student_optimizer, step_size=2, gamma=0.96)    
def train(Teacher_Model, Student_Model,device,iterator, optimizer, criterion): # Define Training Function 
    early_stopping = EarlyStopping(patience=7, verbose=True)
    #print("Training Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    Student_Model.train() # call model object for training 
    for (x1,y) in iterator:
        x,y,y_b=moex(x1, y)
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
        optimizer.zero_grad() # Initialize gredients as zeros 
        KL_Loss=F.kl_div(Predicted_Teacher_Prob, Predicted_Student_Prob)
        #KL_Loss=CKT_Loss(Predicted_Teacher_Prob,Predicted_Student_Prob,y,device)
        #print(x.shape)
        x1=x1.float()
        x1=x1.to(device)
        Predicted_Train_Label=Student_Model(x1)
        Predicted_Train_Label=F.softmax(Predicted_Train_Label,dim=1)
        loss = alpha*IP_Loss+(1-alpha)*KL_Loss
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
        for (x1,y) in iterator:
            x,y,y_b=moex(x1, y)
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
MODEL_SAVE_PATH = os.path.join("/home/mani/Desktop/AK/TASLP 2021/FB Model Training/TASLP_FInal_KD", 'Student_ESC10_CNN.pt') # Define Path to save the model 
Temp=np.zeros([EPOCHS,6]) # Temp Matrix to Store all model accuracy, loss and time parameters 
print("Student ESC10 Model is in Training Mode") 
print("---------------------------------------------------------------------------------------------------------------------")   
early_stopping = EarlyStopping(patience=7, verbose=True) # early Stopping Criteria
for epoch in range(EPOCHS):
    start_time=time.time() # Compute Start Time 
    scheduler.step()
    train_loss, train_acc = train(Teacher_Model, Student_Model,device,train_loader,Student_optimizer, criterion) # Call Training Process 
    train_loss=round(train_loss,2) # Round training loss 
    train_acc=round(train_acc,2) # Round training accuracy 
    valid_loss, valid_acc = evaluate(Teacher_Model,Student_Model,device,valid_loader, criterion) # Call Validation Process 
    valid_loss=round(valid_loss,2) # Round validation loss
    valid_acc=round(valid_acc,2) # Round accuracy 
    end_time=(time.time()-start_time) # Compute End time 
    end_time=round(end_time,2)  # Round End Time 
    print(" | Epoch=",epoch,",| Training Accuracy=",train_acc*100," | Validation Accuracy=",valid_acc*100," | Training Loss=",train_loss," | Validation_Loss=",valid_loss,"Time Taken(Seconds)=",end_time,"|")
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

