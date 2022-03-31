import numpy as np
norm_array = np.array([0,1,2,3,4,5])
_max = 1
_min = -1
norm_array = (norm_array - norm_array.min()) / (norm_array.max()-norm_array.min())
norm_array = norm_array * (_max - _min) + _min
# print(norm_array)
import os


# a = "hello"
# b= "world "
# array = np.array([[1,2],[1,2],[1,2],[1,2],[1,2]])
# num_missing_items = 4
# # padded_array = np.pad(array,(0,num_missing_items),'symmetric')
# # print(padded_array)
# # print(array.shape)
# padded_array = np.zeros((num_missing_items+len(array),2))
# print(padded_array.shape)
# padded_array[:,0] = np.pad(array,(0,num_missing_items),'constant')
# # padded_array[:,1] = np.pad(array,(0,num_missing_items),'constant')
# print(padded_array)

# a  = np.array([0,0])
# for i in range(10):
#     a = np.append(a,np.array([1,2]),axis=0)

# print(a.shape)

a = np.array([-100,50,23,61,55])

a[2:] = 0
print(a)