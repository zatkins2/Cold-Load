import numpy as np

pitch = float(input("pitch (a.u.) = "))
assert pitch >0
r = float(input("radius (a.u.) = "))
assert r > 0
n = int(np.floor(r / pitch))

prev_dim = float(input("previous side length (a.u) = "))
assert prev_dim > 0
n_prev = int(np.floor(prev_dim / (pitch * 2)))

master_array = np.zeros((n,n), dtype = bool)


for row in range(n):

    y = (row + 1) * pitch
    
    for col in range(n):
        
        x = (col + 1) * pitch
        
        if np.sqrt(x**2 + y**2) <= r:

            master_array[row, col] = True

n_true = np.sum(master_array)

print "New number of pyramids = ", 4 * n_true
print "Old number of pyramids = ", 4 * n_prev ** 2
print "% Diff. = ", 100 * float(n_true - n_prev**2) / n_prev ** 2
print master_array
