# AND GATE
import numpy as np

def AND_GATE(x1,x2):
    beta = 0.7
    w1 = 0.5
    w2 = 0.5

    value = w1*x1 + w2*x2
    if value < beta:
        return 0
    else:
        return 1
print("----AND GATE----")    
print(AND_GATE(0,0))
print(AND_GATE(1,0))
print(AND_GATE(0,1))
print(AND_GATE(1,1))

# OR GATE
def OR_GATE(x1,x2):
    beta = 0.4
    w1 = 0.5
    w2 = 0.5

    value = w1*x1 + w2*x2
    if value < beta:
        return 0
    else:
        return 1
print("----OR GATE----")    
print(OR_GATE(0,0))
print(OR_GATE(1,0))
print(OR_GATE(0,1))
print(OR_GATE(1,1))

# NOT AND GATE
def NOT_AND_GATE(x1,x2):
    beta = 0.4
    w1 = 0.5
    w2 = 0.5

    value = w1*x1 + w2*x2
    if value < 0.7:
        return 1
    else:
        return 0
print("----NOT AND GATE----")    
print(NOT_AND_GATE(0,0))
print(NOT_AND_GATE(1,0))
print(NOT_AND_GATE(0,1))
print(NOT_AND_GATE(1,1))

# XOR --0110のようなアウトプット
def XOR_GATE(x1,x2):
    return AND_GATE(
        NOT_AND_GATE(x1,x2),
        OR_GATE(x1,x2)
    )

print("---XOR----")    
print(XOR_GATE(0,0))
print(XOR_GATE(1,0))
print(XOR_GATE(0,1))
print(XOR_GATE(1,1))
