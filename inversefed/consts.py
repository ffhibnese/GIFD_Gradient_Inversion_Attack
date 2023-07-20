"""Setup constants, ymmv."""

PIN_MEMORY = True
NON_BLOCKING = False
BENCHMARK = True
MULTITHREAD_DATAPROCESSING = 4
LAYER_LEN = 8
STYLE_LEN = 18

cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
# cifar10_mean = [0.5, 0.5, 0.5]
# cifar10_std = [0.5, 0.5, 0.5]
cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
mnist_mean = (0.13066373765468597,)
mnist_std = (0.30810782313346863,)
i32_mean = [0.485, 0.456, 0.406]
i32_std = [0.229, 0.224, 0.225]
i64_mean = [0.485, 0.456, 0.406]
i64_std = [0.229, 0.224, 0.225]
i128_mean = [0.485, 0.456, 0.406]
i128_std = [0.229, 0.224, 0.225]
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# imagenet_io_mean = [0.5, 0.5, 0.5]
# imagenet_io_std = [0.5, 0.5, 0.5]
imagenet_io_mean = [0.485, 0.456, 0.406]
imagenet_io_std = [0.229, 0.224, 0.225]

ood_imagenet_mean = [0.485, 0.456, 0.406]
ood_imagenet_std = [0.229, 0.224, 0.225]