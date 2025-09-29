import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#imgs = np.loadtxt("/home/abasit/Downloads/pashto_alphabets.csv")
df = pd.read_csv("pashto_alpha_num_hw.csv")
#df = pd.read_csv("pashto_chrs_printed.csv")
#df = pd.read_csv("pashto_ocr.csv")
#data = pd.read_csv('test.csv')  # path of the .csv file

#for img in range(10):
#    img = 

test=df.iloc[259,1:].values
test=np.reshape(test,(32,32))
print(test.shape)
test=np.array(test,dtype=np.float32)

#
#print(test.shape)
#gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
#
#(thresh, binary) = cv2.threshold(test, 128, 255, cv2.THRESH_BINARY)
#
#print(gray.shape)
##print(binary)
#454
#500
#551
#a = 406
#for i in range(1):  
#    plt.subplot(330 + 1 + i)
#    a = a + 50 
#    img=df.iloc[a,1:].values
#    img=np.reshape(img,(32,32))
#    img=np.array(img,dtype=np.float32)
#    
#    plt.imshow(img, cmap=plt.get_cmap('gray'))
#plt.show()
#
cv2.imshow("WIND",test)
cv2.waitKey(0)

#
###cv2.imread(data.head(0))
###print(data.head(0))
##pix = data.head(0)['alif']
###print(pix)
##im = pix #np.array(pix.split())
##print(im)

