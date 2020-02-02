import numpy as np
from scipy.fftpack import fft, fftshift

def get_features_from_coordinates(coordinates):
    k = 10
    longc = np.max(coordinates.shape)
    meanc = np.mean(coordinates)
    temp_feat = (coordinates - meanc)/longc
    TC = (fft(temp_feat))
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
    return np.abs(coeffs)

def pred_hand(coeffs,_pca,_classifier):
    coeffs_pca = _pca.transform(coeffs)
    #print(coeffs_pca.shape)
    hand_status = _classifier.predict(coeffs_pca)
    return hand_status


def compute_mean_std(loader):
    mean_img = None
    for imgs, _ in loader:
        if mean_img is None:
            mean_img = torch.zeros_like(imgs[0])
        mean_img += imgs.sum(dim=0)
    mean_img /= len(loader.dataset)

    std_img = torch.zeros_like(mean_img)
    for imgs, _ in loader:
        std_img += ((imgs - mean_img)**2).sum(dim=0)
    std_img /= len(loader.dataset)
    std_img = torch.sqrt(std_img)
    std_img[std_img == 0] = 1

    return mean_img, std_img