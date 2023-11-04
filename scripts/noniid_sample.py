import random
import numpy as np

seed = 211
random.seed(seed)
np.random.seed(seed)
N = 18000
ratio = 0.2
size1 = int(N*ratio)
N = N - size1
min_samples = 50
max_samples = size1
min_size = 0
max_size = np.inf
alpha = 0.1
n_clients = 10
repeat_times = 0

while(min_size < min_samples or max_size > max_samples):
    repeat_times += 1
    proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
    sizes = N*proportions
    min_size = min(sizes)
    max_size = max(sizes)

print(repeat_times)
print(size1)
print(sizes)