import numpy as np
import sys,os
sys.path.append(os.pardir)
sys.path.append("/Users/kenjikaminogo/Desktop/python/01_deeplearning/DeepLearningFromScratch_01/common/")
from util import im2col

x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1,5,5,stride=1,pad=0)
print(col1.shape)