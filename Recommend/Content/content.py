import tensorflow as tf
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler


Types = 18
Type_Dict = {
    'Action':0,
    'Adventure':1,
    'Animation':2,
    'Children\'s':3,
    'Comedy':4,
    'Crime':5,
    'Documentary':6,
    'Drama':7,
    'Fantasy':8,
    'Film-Noir':9,
    'Horror':10,
    'Musical':11,
    'Mystery':12,
    'Romance':13,
    'Sci-Fi':14,
    'Thriller':15,
    'War':16,
    'Western':17
}
scaler_User, scaler_Item, scaler_Rating = None, None, None

# content-based filtering with neural network
def Init_User():
    users = []
    with open('users.dat') as f:
        reader = list(f.readlines())
        for i in reader:
            user = i.split('::')
            if user[1] == 'M':
                user[1] = 1
            else:
                user[1] = 0
            user = user[1:4] 
            user = [float(i) for i in user]
            users.append(user)
    f.close()
    users = np.array(users) 
    
    global scaler_User
    scaler_User = MinMaxScaler()
    users = scaler_User.fit_transform(users)
    return users   
    
def Init_Item(M = 4000):
    items = np.zeros((M,Types))
    with open('movies.dat',errors='ignore') as f:
        reader = list(f.readlines())
        for i in reader:
            item = i.split('::')
            for j in item[2].split('|'):
                items[int(item[0])][Type_Dict[j.strip()]] = 1 
    f.close()
    
    global scaler_Item
    scaler_Item = MinMaxScaler()
    items = scaler_Item.fit_transform(items)
    
    return items

def Init_Rating(N = 6040,M = 4000):
    Rating = np.zeros((N,M))
    with open('ratings.dat') as f:
        reader = list(f.readlines())
        ratings = []
        for i in reader:
            data = i.split('::')
            Rating[int(data[0])-1,int(data[1])-1] = float(data[2])
    f.close()
    datas = []
    for i in Rating:
        if np.sum(i) != 0:
            datas.append(i)
    Rating = np.array(datas)
    
    global scaler_Rating
    scaler_Rating = MinMaxScaler()
    Rating = scaler_Rating.fit_transform(Rating)
    
    return Rating

def Main():
    # Build the User Model and Item Model
    User_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(32)
    ])
    
    Item_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(32)
    ])
    
    # Create the User input and point to the base network
    Users = Init_User()
    print(Users.shape)
    User_Input = tf.keras.layers.Input(shape=(Users.shape[1]))
    Vu = User_NN(User_Input)
    Vu = tf.linalg.l2_normalize(Vu,axis=1)
    
    # Create the Item input and point to the base network
    Items = Init_Item()
    print(Items.shape)
    Item_Input = tf.keras.layers.Input(shape=(Items.shape[1]))
    Vm = Item_NN(Item_Input)
    Vm = tf.linalg.l2_normalize(Vm,axis=1)
    
    # Measure the similarity between user and item
    Output = tf.keras.layers.Dot(axes=1)([Vu,Vm])
    
    # specify the model
    model = tf.keras.models.Model(inputs=[User_Input,Item_Input],outputs=Output)
    print(model.summary())
    
    # train the model
    Rating = Init_Rating()
    # print(Rating)
    tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer='adam',loss="mse")
    
    id_User = range(Users.shape[0])
    id_Item = range(Items.shape[0])
    Case = 1000
    Sample = 1000
    for i in range(Case):
        print()
        print(f"Case {i+1}:")
        id_User_Sample = random.sample(id_User,Sample)
        id_Item_Sample = random.sample(id_Item,Sample)
        Rating_Sample = [Rating[id_User_Sample[i],id_Item_Sample[i]] for i in range(Sample)]
        for j in range(Sample):
            while Rating_Sample[j] == 0:
                id_User_Sample[j] = random.choice(id_User)
                id_Item_Sample[j] = random.choice(id_Item)
                Rating_Sample[j] = Rating[id_User_Sample[j],id_Item_Sample[j]]
                
        model.fit([Users[id_User_Sample],Items[id_Item_Sample]],np.array(Rating_Sample).reshape(-1,1),epochs=100) 
    
    model.save('content.h5')
    
    cnt = 0
    for i in id_User:
        for j in id_Item:
            if Rating[i][j] != 0:
                rating = model.predict([Users[i].reshape(1,-1),Items[j].reshape(1,-1)])
                if abs(rating - Rating[i][j]) <= 0.1:
                    cnt += 1  
    
    print(f"Accuracy: %.2lf %%"%(cnt/np.sum(Rating!=0)*100))
    
if __name__ == '__main__':
    Main()