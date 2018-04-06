



# In[3]:
import matplotlib.pyplot as plt


# In[10]:


def show_train_history1(train_history1,train_history2,train_history3, train, validation):
    plt.plot(train_history1[train])
    plt.plot(train_history1[validation])
    plt.plot(train_history2[train])
    plt.plot(train_history2[validation])
    plt.plot(train_history3[train])
    plt.plot(train_history3[validation])
    plt.title("Train History")
    plt.ylabel("train")
    plt.xlabel("Epoch")
    #plt.legend(["CNN1", "CNN1_test","CNN2", "CNN2_test","CNN4", "CNN4_test"], loc="center right")
    plt.legend(['shallow_train_acc', 'shallow_test_acc', 'mid_train_acc', 'mid_test_acc', 'deep_train_acc', 'deep_test_acc'], loc="best")
    plt.show()

def show_train_history2(train_history1,train_history2,train_history3, train, validation):
    plt.plot(train_history1[train])
    plt.plot(train_history1[validation])
    plt.plot(train_history2[train])
    plt.plot(train_history2[validation])
    plt.plot(train_history3[train])
    plt.plot(train_history3[validation])
    plt.title("Train History")
    plt.ylabel("train")
    plt.xlabel("Epoch")
    #plt.legend(["CNN1", "CNN1_test","CNN2", "CNN2_test","CNN4", "CNN4_test"], loc="center right")
    plt.legend(['shallow_train_loss', 'shallow_test_loss', 'mid_train_loss', 'mid_test_loss', 'deep_train_loss', 'deep_test_loss'], loc="best")
    plt.show()

import pickle
with open('deep1', 'rb') as file:
    h2 =pickle.load(file)
with open('deep2', 'rb') as file:
    h1 =pickle.load(file)
with open('deep3', 'rb') as file:
    h3 =pickle.load(file)



# In[12]:


show_train_history1(h1,h2,h3, "acc","val_acc")


# In[9]:


# In[13]:


show_train_history2(h1,h2,h3, "loss","val_loss")

