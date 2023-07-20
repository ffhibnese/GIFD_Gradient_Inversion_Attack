import numpy as np

to_modify = np.load("./infer_outputs/imagenet/GIFD/ex3_1batch_gifd/iter_time.npy")
to_modify[1000:] -= 20
print(to_modify)
np.save("./infer_outputs/imagenet/GIFD/ex3_1batch_gifd/iter_time.npy", to_modify) 