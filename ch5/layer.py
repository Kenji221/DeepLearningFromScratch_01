import numpy as np
# backwordの呼び出し時に最初に定義した値を改めて呼び出すから必要
class MultLayer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self,x,y):
        self.x = x
        self.y = y
        output = self.x * self.y

        return output
    
    def backward(self,d_out):
        dx = self.y * d_out
        dy = self.x * d_out

        return dx, dy
# backwaordは積と違ってただ１をかけているだけだから最初のinitでの定義は不要となる
class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        return x + y 
    
    def backward(self, d_out):
        dx = 1 * d_out
        dy = 1 * d_out
        return dx,dy
    
# 例題
apple_count = 2
apple_price = 100
orange_count = 3
orange_price = 150
tax = 1.1

apple_Layer = MultLayer()
orange_Layer = MultLayer()
apple_orange_Layer = AddLayer()
tax_Layer = MultLayer()

# 値段の計算
apple_amount = apple_Layer.forward(apple_count,apple_price)
orange_amount = orange_Layer.forward(orange_count,orange_price)
apple_orange_amount = apple_orange_Layer.forward(apple_amount,orange_amount)
total_amount = tax_Layer.forward(apple_orange_amount,tax)

print(apple_amount)
print(orange_amount)
print(apple_orange_amount)
print(total_amount)

# 逆伝搬の計算
print("Backward total_amount",tax_Layer.backward(1))
