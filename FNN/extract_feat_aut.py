import numpy as np
from scipy.fftpack import fft, fftshift
from matplotlib import pyplot as plt
import sys
import os

files = []
for i,arg in enumerate(sys.argv):
    if(i>0):
        if(not os.path.isfile(arg)):
            print("{} does not exist".format(arg))
        else:
            files.append(arg)

for f in files:
    arr = np.loadtxt(f)
    z_arr = arr[:,0] + 1j*arr[:,1]
    k = 10
    ind_arr = np.where(z_arr == -1 -1j)[0]
    count = 0
    start = 0
    f_arr = open('{}_feat.csv'.format(os.path.splitext(f)[0]),'w')
    for i in ind_arr:
        count = count + 1
        print(i)
        temp_arr = z_arr[start:i]
        longc = np.max(temp_arr.shape)
        meanc = np.mean(temp_arr)
        temp_arr = (temp_arr - meanc)/longc
        start = i+1
        TC = (fft(temp_arr))
        coeffs = np.zeros(2*k + 1,dtype=complex)
        coeffs[2*k+1-k-1:2*k+1] = TC[0:k+1]
        coeffs[:k] = TC[2*k+1 - k:2*k+1]
        if(coeffs[k+1] < coeffs[k-1]):
            coeffs = np.flip(coeffs)
        phi = np.angle(coeffs[k+1]*coeffs[k-1])
        coeffs = coeffs*np.exp(-1j*phi)
        theta = np.angle(coeffs[k+1])
        temp_exp = np.exp(-1j*np.arange(-k,k+1)*theta)
        coeffs = coeffs*temp_exp
        coeffs=coeffs/np.abs(coeffs[k+1])
        for _i,c in enumerate(coeffs):
            if(_i == 20):
                f_arr.write(str(np.abs(c)))
            else:
                f_arr.write(str(np.abs(c))+",")
        f_arr.write("\n")
    f_arr.close()
        