import pickle
import numpy as np
#from skimage.util import view_as_windows
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold,cross_val_score,cross_val_predict,train_test_split,cross_validate
import itertools


data = []
tag_l= []
tag_s= set()

with open('Brown_train.txt','r') as fi:
    templ = []
    tempt = []
    for ln in fi.readlines():
        words = ln.split(" ")
        for i in range(0,len(words)-1):
            tag = words[i].split("/")
            templ.append(tag[0])
            tempt.append(tag[1])         
            tag_s.add(tag[1])
        data.append(templ)
        tag_l.append(tempt)
        templ = []
        tempt = []

glove = pickle.load(open('glove_50.pkl','rb'))
data_train,data_test,tag_train,tag_test = train_test_split(data,tag_l,train_size = 0.7,test_size = 0.3)
print(len(data_test))
datav = []
for i in data_train:
    tempv = []
    for j in i:
        try:
            tempv.extend(glove[j])
        except:
            tempv.extend([-1]*50)
    datav.append(tempv)

datavp = []
for x in datav:
    px = [-1]*100+x+[-1]*100
    datavp.append(px)
    
#tag_s = list(set(tag_l))
tag_s = list(tag_s)
tag_v = []

for x in tag_train:
    tempv = []
    for y in x:
        tempt = [0]*len(tag_s)
        tempt[tag_s.index(y)] = 1
        tempv.append(tempt)
    tag_v.extend(tempv)

data = []
for ln in datavp:
    for i in range(100,len(ln)-100,50):
        data.append(ln[i-100:i+150])

data_train1 = np.array(data)
tag_train1 = np.array(tag_v)


datav = []
for i in data_test:
    tempv = []
    for j in i:
        try:
            tempv.extend(glove[j])
        except:
            tempv.extend([-1]*50)
    datav.append(tempv)

datavp = []
for x in datav:
    px = [-1]*100+x+[-1]*100
    datavp.append(px)
    
#tag_s = list(set(tag_l))
tag_s = list(tag_s)
tag_v = []

for x in tag_test:
    tempv = []
    for y in x:
        tempt = [0]*len(tag_s)
        tempt[tag_s.index(y)] = 1
        tempv.append(tempt)
    tag_v.extend(tempv)

data = []
for ln in datavp:
    for i in range(100,len(ln)-100,50):
        data.append(ln[i-100:i+150])

data_test2 =np.array(data)
tag_test2 = np.array(tag_v)

#print(data.shape,tag_v.shape)
from keras.models import Sequential
from keras.layers import *

model = Sequential()
model.add(Dense(256,activation='relu',input_shape=(250,)))
model.add(Dense(128,activation='relu'))
model.add(Dense(len(tag_s),activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy')
model.summary()


model.fit(data_train1,tag_train1,validation_data=[data_test2,tag_test2],epochs=10)

pred_tag = model.predict(data_test2)
pred_tag = pred_tag.argmax(-1)

tag_test2 = tag_test2.argmax(-1)
pred_tag_o = []
test_tag_o = []
for i in range(len(pred_tag)):
    pred_tag_o.append(tag_s[pred_tag[i]])
    test_tag_o.append(tag_s[tag_test2[i]])


#print(tag_test)
print(classification_report(test_tag_o,pred_tag_o))
print(pred_tag_o[0])
print(pred_tag_o[1])


f = open("pred.txt","w")
data_test = list(itertools.chain(*data_test))
map(lambda x:[x],data_test)
tag_test = list(itertools.chain(*tag_test))
map(lambda x:[x],tag_test)
#pred_tag_o = list(itertools.chain(*pred_tag_o))
#map(lambda x:[x],pred_tag_o)
#print(pred_tag_o[0])
#print(len(tag_test))
#print(len(test_tag_o))
string = ""
for z in range(len(data_test)):
	string=data_test[z]+"\t"+tag_test[z]+"\t"+pred_tag_o[z]
	f.write(string)
	f.write("\n")
