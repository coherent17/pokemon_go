import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

class KNN():
    def __init__(self):
        pass

    def data_preprocessing(self):
        #DATA is the list that store the whole dataset
        DATA=np.genfromtxt("Pokemon.csv",delimiter=',',encoding="utf-8",dtype=str)

        #read feature data as the float
        dataX=np.zeros((DATA.shape[0]-1,DATA.shape[1]-5),dtype=float)
        for i in range(DATA.shape[0]-1):
            for j in range(DATA.shape[1]-5):
                dataX[i,j]=float(DATA[i+1,j+4])

        #print(dataX.shape)  (1072, 7)
    
        #read the target data and convert true as 1 false as 0
        dataT=np.zeros(len(DATA)-1,dtype=float)
        for i in range(len(DATA)-1):
            if DATA[i+1,-1]=='FALSE':
                dataT[i]='0'
            else:
                dataT[i]='1'

        #read the name of the pokemon and store in the data_name[]
        data_name=[]
        for i in range(len(DATA)-1):
           data_name.append(DATA[i+1,1])

        #print(len(data_name))  #1072
        return dataX,dataT,data_name 

    def normalize(self,X):
        mean_X=[]
        std_X=[]
        for i in range(X.shape[1]):
            mean_X.append(np.mean(X[:,i]))
            std_X.append(np.std(X[:,i]))
        X_n=np.zeros(np.shape(X))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_n[i,j]=(X[i,j]-mean_X[j])/std_X[j]
        return X_n

    def train_test_split(self,dataX,dataT):
        dataX_train=dataX[0:750,:] #(750,7))
        dataX_test=dataX[750:,:] #(322,7)
        dataT_train=dataT[0:750].reshape(750,1) #(750,1)
        dataT_test=dataT[750:].reshape(322,1) #(322,1)
        return dataX_train,dataX_test,dataT_train,dataT_test

    def KNN_score(self,dataX_train,dataX_test,dataT_train,dataT_test):
        accuracy_store=[] #store the accuracy of different K from 1 ~ 10
        for j in range(1,11): #change the value of K
            result=[]
            for i in range(np.shape(dataX_test)[0]): #change the row of the testing data
                distance=KNN.euclidean_distance(dataX_train, dataX_test[i,:])
                result.append(KNN.vote(distance,j,dataT_train))
            accuracy_store.append(KNN.accuracy(result, dataT_test))
        k=np.argsort(accuracy_store)[-1] #the K value corresponding to the highest accuracy
        print('the maximum accuracy happen at K= %d is' %(k) ,accuracy_store[k])
        print(accuracy_store)
        return accuracy_store,k

    #per row of the testing data return a list of the euclidean distance
    def euclidean_distance(self,X_train,X_test):
        distance=[]
        for i in range(np.shape(X_train)[0]):
            dis=0
            for j in range(np.shape(X_train)[1]):
                dis+=(X_train[i,j]-X_test[j])**2
            distance.append(np.sqrt(dis))
        return distance

    def vote(self,distance,k,dataT_train):
        ind_sort=np.argsort(distance)
        ind=[]
        for i in range(k):
            ind.append(ind_sort[i])
        flag=[0,0] #not_legend,legend
        for i in dataT_train[ind]:
            if i==0: # not_legend
                flag[0]+=1
            elif i==1: #legend
                flag[1]+=1 
        big=np.max(flag)
        result=[]
        if flag[0]==big:
            result='not_legend'
        elif flag[1]==big:
            result='legend'
        return result

    def accuracy(self,result,dataT_test):
        flag=0
        for i in range(len(result)):
            if result[i]=='not_legend' and dataT_test[i]==0:
                flag+=1
            elif result[i]=='legend' and dataT_test[i]==1:
                flag+=1
        return flag/len(result)
    
    def KNN_classfier(self,index,dataX_train,dataX_test,dataT_train,dataT_test,k): #use the highest k to classfier the data
        distance=KNN.euclidean_distance(dataX_train, dataX_test[0,:])
        result=(KNN.vote(distance,k,dataT_train))
        print(a,': the prediction type: '+result)
        if result=='not_legend' and dataT_test[index]==0:
            print('the prediction is correct!')
        elif result=='legend' and dataT_test[index]==1:
            print('the prediction is correct!')
        else:
            print('the prediction is wrong!')
            if dataT_test[index]==0:
                print('the correct answer is not_legend')
            elif dataT_test[index]==1:
                print('the correct answer is legend')
        return result

KNN=KNN()
dataX,dataT,data_name=KNN.data_preprocessing()
dataX_train,dataX_test,dataT_train,dataT_test=KNN.train_test_split(dataX, dataT)
dataX_train=KNN.normalize(dataX_train)
dataX_test=KNN.normalize(dataX_test)
accuracy_store,k=KNN.KNN_score(dataX_train, dataX_test, dataT_train, dataT_test)
a=np.random.randint(np.shape(dataX_test)[0])
result=KNN.KNN_classfier(a,dataX_train, dataX_test[a,:].reshape(1,7), dataT_train, dataT_test,k)
a=np.random.randint(np.shape(dataX_test)[0])
result=KNN.KNN_classfier(a,dataX_train, dataX_test[a,:].reshape(1,7), dataT_train, dataT_test,k)
a=np.random.randint(np.shape(dataX_test)[0])
result=KNN.KNN_classfier(a,dataX_train, dataX_test[a,:].reshape(1,7), dataT_train, dataT_test,k)

#visualize
x=np.linspace(1, 10,10)
plt.plot(x,accuracy_store)
plt.xlabel('K')
plt.ylabel('accuracy')
plt.title('accuracy versus the value of K')
plt.grid(True)
plt.show()