from AI import selftrain,play
import random
class Human():
    def choice(self):
        ch = input('input your choice: (0 Silent / 1 Confess)')
        return int(ch)
    def update(self,action):
        return

class Player1():
    # choose to kepp Silent until get 5 hurts
    count = 0
    def choice(self):
        return 0 if self.count<5 else 1
    def update(self,action):
        if action == 1:
            self.count += 1
            
class Player2():
    # choose to simulate the other one
    last = int(random.random()*2)
    def choice(self):
        return self.last
    
    def update(self,action):
        self.last = action
        
N = int(input("Number of games to train: "))
ai = selftrain(N)
# # me = Human()
# for i in range(N):
#     print('Game ',i+1)
#     P1 = Player1()
#     # P2 = Player2()
#     play(ai,P1)