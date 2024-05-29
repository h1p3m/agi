import torch
import gc 
from numba import cuda
 
gc.collect()

torch.cuda.empty_cache() 


cuda.select_device(0) # choosing second GPU 
cuda.close()