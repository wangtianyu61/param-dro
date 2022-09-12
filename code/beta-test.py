import numpy as np


sample_num = 5000000


#the center is empirical distribution
def chi2_distance(p, q):
    div = 0
    for a, b in zip(p, q):
        if b != 0:
            div += (a-b)**2/a
    return div

def KL(p, q):
    pass

def freq_compute(sample, region):
    freq = np.zeros(len(region))
    for element in sample:
        #return the index i such that a[i - 1] <= v < a[i]
        idx = np.searchsorted(region, element, side = 'right')
        freq[idx - 1] += 1
    return freq[1:]/sum(freq[1:])
        
bin_num = 100
if __name__ == '__main__':
    sample_1 = np.random.beta(3, 1, size = sample_num)
    sample_2 = np.random.beta(2, 1, size = sample_num)
    #region = np.sort(np.array([0] + list(set(sample_1))))
    region = np.array(range(bin_num))/bin_num
    p = freq_compute(sample_1, region)
    q = freq_compute(sample_2, region)
    print(chi2_distance(p, q))
        

