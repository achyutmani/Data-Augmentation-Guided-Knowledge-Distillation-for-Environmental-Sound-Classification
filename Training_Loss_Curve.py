import numpy as np
Data=np.load('Teacher_ESC10_CNN_Parameters.npy')
print(np.shape(Data))
#import xlrd
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.style as style 
style.use('seaborn-poster') #sets the size of the charts
style.use('seaborn-whitegrid')
#style.available
#print(Train_Accuracy)
#Train_Accuracy=Data[:,3]
#print(Train_Accuracy)
#Validation_Accuracy=Data[:,4]
Epochs=24
Train_Loss=Data[:,3]
Valid_Loss=Data[:,4]
#print(len(Train_Loss))
plt.xticks(fontsize=28) 
plt.yticks(fontsize=28) 
#plt.xticks(np.arange(0, 32, 1),fontsize=28) 
#plt.yticks(np.arange(0, 1, 0.1),fontsize=28) 
plt.axis([0,Epochs, 0,2.5])
plt.plot(Train_Loss, label='Training Loss',color="blue",linestyle='--',linewidth=5)
plt.plot(Valid_Loss, label='Validation Loss',color="red",linestyle='--',linewidth=5)
plt.legend(loc='upper right',fontsize=28)
plt.xlabel('Number of Epochs', fontsize=28) 
#plt.ylabel('Accuracy(%)', fontsize=28)
plt.ylabel('Loss', fontsize=28)
plt.title('Training and Validation Losses',fontsize=28)
plt.savefig('ESC10_Teacher_Train_Val_Loss.png')
plt.show()
