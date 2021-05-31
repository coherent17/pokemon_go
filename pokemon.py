import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

class KNN():
    def __init__(self):
        pass
 
    def data_preprocessing(self):
        #DATA is the list that store the whole dataset
        data_org=np.genfromtxt("Pokemon.csv",delimiter=',',encoding="utf-8",dtype=str)

        #delete the title of the dataset
        DATA=np.delete(data_org,[0],axis=0)

        #shuffle the data to avoid the strange
        np.random.shuffle(DATA)
        

        #read feature data as the float
        dataX=np.zeros((DATA.shape[0],7),dtype=float)
        for i in range(DATA.shape[0]):
            for j in range(7):
                dataX[i,j]=float(DATA[i,j+4])

        #print(dataX.shape)  #(1072, 7)
    
        #read the target data and convert true as 1 false as 0
        dataT=np.zeros(len(DATA),dtype=float)
        for i in range(len(DATA)):
            if DATA[i,-1]=='FALSE':
                dataT[i]='0'
            else:
                dataT[i]='1'
        #print(dataT.shape) #(1072,)
        #read the name of the pokemon and store in the data_name[]
        data_name=[]
        for i in range(len(DATA)):
           data_name.append(DATA[i,1])

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
        dataX_train=dataX[0:643,:] #(643,7)
        dataX_valid=dataX[643:857,:] #(214,7)
        dataX_test=dataX[857:,:] #(215,7)
        
        dataT_train=dataT[0:643].reshape(643,1) #(643,1)
        dataT_valid=dataT[643:857].reshape(214,1) #(214,1)
        dataT_test=dataT[857:].reshape(215,1) #(215,1)
        return dataX_train,dataX_valid,dataX_test,dataT_train,dataT_valid,dataT_test

    def KNN_score(self,dataX_train,dataX_valid,dataT_train,dataT_valid):
        accuracy_store=[] #store the accuracy of different K from 1 ~ 10
        for j in range(1,11): #change the value of K
            result=[]
            for i in range(np.shape(dataX_valid)[0]): #change the row of the testing data
                distance=KNN.euclidean_distance(dataX_train, dataX_valid[i,:])
                result.append(KNN.vote(distance,j,dataT_train))
            accuracy_store.append(KNN.accuracy(result, dataT_valid))
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

    def accuracy(self,result,dataT_valid):
        flag=0
        for i in range(len(result)):
            if result[i]=='not_legend' and dataT_valid[i]==0:
                flag+=1
            elif result[i]=='legend' and dataT_valid[i]==1:
                flag+=1
        return flag/len(result)
    
    def KNN_classfier(self,index,dataX_train,dataX_test,dataT_train,dataT_test,k): #use the highest k to classfier the data
        distance=KNN.euclidean_distance(dataX_train, dataX_test[0,:])
        result=(KNN.vote(distance,k,dataT_train))
        print('The prediction type of',a,data_name[a],":",result)
        if result=='not_legend' and dataT_test[index]==0:
            print('The prediction is correct!')
        elif result=='legend' and dataT_test[index]==1:
            print('The prediction is correct!')
        else:
            print('The prediction is wrong!')
            if dataT_test[index]==0:
                print('the correct answer is not_legend')
            elif dataT_test[index]==1:
                print('the correct answer is legend')
        return result

    def qm_value_calculater(self,dataX,data_name):
        #parameter
        max_hp=np.max(dataX[:,1])
        max_attack=np.max(dataX[:,2])
        max_defense=np.max(dataX[:,3])
        max_speed=np.max(dataX[:,6])
        #some noise
        a=np.random.randint(-3,3)
        b=np.random.randint(-3,3)
        c=np.random.randint(-3,3)
        d=np.random.randint(-3,3)
        qm_value_store=[]
        for j in data_name:
            for i in range(len(data_name)):
                if data_name[i]==j:
                    qm_value=((dataX[i,1]+a)/max_hp)*(((dataX[i,2]+b)/max_attack)**2)*(((dataX[i,3]+c)/max_defense)*((dataX[i,6]+d)/max_speed))**(1/2)
                    qm_value_store.append(qm_value)
        qm_max=np.max(qm_value_store)
        for i in range(len(qm_value_store)):
            qm_value_store[i]=(qm_value_store[i]/qm_max)*100
        return qm_value_store

    # def qm_safari(self,k,dataX,dataT,dataX_train,dataT_train):
    #     name=input("Please enter the Pokemon you want to predict:")
    #     for i in range(len(data_name)):
    #         if data_name[i]==name:
    #             KNN.KNN_classfier(i,dataX_train, dataX[i,:].reshape(1,7), dataT_train, dataT,k)




KNN=KNN()
dataX,dataT,data_name=KNN.data_preprocessing()
dataX_train,dataX_valid,dataX_test,dataT_train,dataT_valid,dataT_test=KNN.train_test_split(dataX, dataT)
dataX_train=KNN.normalize(dataX_train)
dataX_valid=KNN.normalize(dataX_valid)
dataX_test=KNN.normalize(dataX_test)
accuracy_store,k=KNN.KNN_score(dataX_train, dataX_valid, dataT_train, dataT_valid)
a=np.random.randint(np.shape(dataX_test)[0])
result=KNN.KNN_classfier(a,dataX_train, dataX_test[a,:].reshape(1,7), dataT_train, dataT_test,k)
a=np.random.randint(np.shape(dataX_test)[0])
result=KNN.KNN_classfier(a,dataX_train, dataX_test[a,:].reshape(1,7), dataT_train, dataT_test,k)
a=np.random.randint(np.shape(dataX_test)[0])
result=KNN.KNN_classfier(a,dataX_train, dataX_test[a,:].reshape(1,7), dataT_train, dataT_test,k)
# KNN.qm_safari(k,dataX,dataT,dataX_train,dataT_train)
qm_value_store=KNN.qm_value_calculater(dataX,data_name)

# visualize
x=np.linspace(1, 10,10)
plt.plot(x,accuracy_store)
plt.xlabel('K')
plt.ylabel('accuracy')
plt.title('accuracy versus the value of K')
plt.grid(True)
plt.show()