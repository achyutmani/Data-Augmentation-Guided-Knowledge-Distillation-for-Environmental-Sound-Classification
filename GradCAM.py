import torch.optim as optim # Import optim 
import torchvision # Import torchvision 
import h5py #import h5py
from torch.utils.data import dataset # import dataset
from ESC10Customdataset_Train import LAEData_Train # Call Customdataloader to read Training Data
from ESC10Customdataset_Test import LAEData_Test # Call Customdataloader to read Test Data
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
#from Attention import CAM_Module # import channel attention module
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
from scipy import ndimage, misc
SEED = 1234 # Initialize seed 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
import librosa.display
device = torch.device('cuda') # Define device type 
num_classes=10 # Define Number of classes 
in_channel=1   # Define Number of Input Channels 
learning_rate=2e-4 # Define Learning rate 
batch_size=64 # Define Batch Size 
EPOCHS =1000   # Define maximum Number of Epochs
FC_Size=512
SFC_Size=512
Temp=3
alpha=0.7
N_models=6
warnings.filterwarnings("ignore")
train_transformations = transforms.Compose([ # Training Transform 
    #transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])
test_transformations = transforms.Compose([ # Test Transform 
    #transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])
train_dataset=LAEData_Train(transform=train_transformations) # Create tensor of training data 
Test_Dataset=LAEData_Test(transform=test_transformations)# Create tensor of test dataset 
train_size = int(0.7 * len(train_dataset)) # Compute size of training data using (70% As Training and 30% As Validation)
valid_size = len(train_dataset) - train_size # Compute size of validation data using (70% As Training and 30% As Validation)
Train_Dataset,Valid_Dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size]) # Training and Validation Data After (70%-30%)Data Split 
#train_set,test_set=torch.utils.data.random_split(dataset,[6000,2639])
#Labels=pd.read_csv("Devlopment.csv")
train_loader=DataLoader(dataset=Train_Dataset,batch_size=batch_size,shuffle=True) # Create Training Dataloader 
valid_loader=DataLoader(dataset=Valid_Dataset,batch_size=batch_size,shuffle=False)# Create Validation Dataloader 
test_loader=DataLoader(dataset=Test_Dataset,batch_size=batch_size,shuffle=False) # Create Test Dataloader 
class Teacher(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(Teacher, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2])
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
        #x1=self.PAM(x)
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
#print(Teacher_Model)
Teacher_Model=Teacher_Model.to(device)
Teacher_optimizer = optim.Adam(Teacher_Model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function 
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
MODEL_SAVE_PATH = os.path.join("/home/mani/Desktop/AK/TASLP 2021/FB Model Training", 'PAM_ESC10_CNN.pt') # Define Path to save the model 
Teacher_Model.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_loss, test_acc = evaluate(Teacher_Model, device, test_loader, criterion)
test_loss=round(test_loss,2)
test_acc=round(test_acc,2)
print("|Test Loss=",test_loss,"Test Accuracy=",test_acc*100)
#print(Teacher_Model.PAM)
def feature_Map(model,Input_Data):
    class SaveOutput:
        def __init__(self):
            self.outputs = []
            
        def __call__(self, module, module_in, module_out):
            self.outputs.append(module_out) 
        def clear(self):
            self.outputs = [] 
    save_output= SaveOutput()
    hook_handles=[]
    handle1 = model.PAM.register_forward_hook(save_output)
    hook_handles.append(handle1)
    Temp= model(Input_Data)
    def module_output_to_numpy(tensor):
        return tensor.detach().to('cpu').numpy()
    F_Map = module_output_to_numpy(save_output.outputs[0])
    F_Map=torch.from_numpy(F_Map).to(device)    
    return F_Map
#key ='BABY_CRY_4-167077-C.ogg'
#key= 'CHAINSAW_3-118972-B.ogg'
#key='CLOCK_TICK_2-88724-A.ogg'    
#key='DOGBARK_3-144028-A.ogg'
#key='FIRECRACKING_1-17808-A.ogg'
#key='HELICOPTER_5-177957-E.ogg'
#key='PSNEEZE_4-156843-A.ogg'
#key='RAIN_2-117625-A.ogg'
#key='ROOSTER_2-100786-A.ogg'
key='SEA_Waves_4-167063-B.ogg'
with h5py.File('ESC10.hdf5', 'r') as f: # Database for Left Channel Test Spectrogram 
            SG_Data = f[key][()]
            SG_Data1=np.array(SG_Data)
Data=test_transformations(SG_Data1).unsqueeze(0)               
#Data=SG_Data.unsqueeze(0)             
#Data=torch.randn((1,1,128,461))   
Attention_Data= feature_Map(Teacher_Model,Data.to(device)).squeeze(0).cpu()
#print(Attention_Data.shape)
Attention_Data=np.transpose(Attention_Data.cpu(),(1,2,0))
#print(Attention_Data.shape)
Weight_Layer=Teacher_Model.fc.weight.cpu()
#print(Weight_Layer.shape)
# Predicting the class label 
Predicted_Class_Label=torch.softmax(Teacher_Model(Data.to(device)),dim=1)
#print(Predicted_Class_Label)
Max_Prob=Predicted_Class_Label.max(1, keepdim=True)[1].cpu()
print(Max_Prob)
CAM_Weight=Weight_Layer[Max_Prob,:].squeeze(0).squeeze(0)
#print(CAM_Weight.shape)
cam = np.dot(Attention_Data.detach(),CAM_Weight.detach())
#print(cam.shape)
import matplotlib.pyplot as plt
Test_Data=Data.squeeze(0).squeeze(0)
class_activation = ndimage.zoom(cam, zoom=(32,31),order=2)
#print(class_activation.shape)
#print(SG_Data.shape)
#plt.imshow(Test_Data, cmap='jet',alpha=1)
#plt.imshow(class_activation,cmap='jet',alpha=0.1)
#print(SG_Data.shape)
plt.subplot(1,2,1)
librosa.display.specshow(SG_Data, sr=44100,y_axis='linear',cmap='coolwarm');
plt.ylabel("Frequency",size=25)
plt.xlabel("Time",size=25)
plt.title('Spectrogram Image')
cbar1=plt.colorbar(orientation="horizontal", pad=0.1)
cbar1.ax.set_ylabel('Image Pixels', rotation=90)
plt.xticks(size=25)
plt.yticks(size=25)
plt.subplot(1,2,2)
librosa.display.specshow(class_activation[:,0:431],sr=44100,y_axis='linear',cmap='coolwarm');
plt.ylabel("Frequency",size=25)
plt.xlabel("Time",size=25)
plt.title('Atttention Feature Map')
cbar2=plt.colorbar(orientation="horizontal", pad=0.1)
cbar2.ax.set_ylabel('Attention Weights', rotation=90)
plt.xticks(size=25)
plt.yticks(size=25)
figure = plt.gcf()  # get current figure
figure.set_size_inches(25,10) # set figure's size manually to your full screen (32x18)
plt.savefig('Seawaves_Attention_Map_New.png', bbox_inches='tight') # bbox_inches removes extra white spaces
plt.show()
# plt.imshow(SG_Data,cmap='coolwarm',alpha=1)
# plt.imshow(class_activation[:,0:431],cmap='jet',alpha=8.5)
# plt.show()
