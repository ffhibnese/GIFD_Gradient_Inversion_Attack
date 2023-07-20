export CUDA_VISIBLE_DEVICES=0

dataset="FFHQ"
# dataset="CIFAR10"
model="ResNet18"
init='randn'
cost_fn='sim_cmpr0.95'
indices='def'
weights='equal'
# weights='linear'
# weights='exp'
filter='none'
# filter='median'


lr=1e-1
gias_lr=1e-3

checkpoint_path='results/G_b6dfe5d78698627ab719fd4a6f74aae2.pkl'

max_iterations=1000
gias_iterations=5000
num_exp=10
num_images=1
restarts=4

# Priors
tv=1e-6
bn_stat=0
image_norm=0
group_lazy=0

# generative_model='stylegan2'
# gen_dataset='I512'
generative_model='DCGAN'
gen_dataset='C10'

target_id=10000
# data_path='/home/jjw/.torch/ILSVRC2012'

data_path='~/.torch/FFHQ'

# data_path='~/.torch/'


python -u rec_mult.py \
    --unsigned --boxed --save_image \
    --cost_fn $cost_fn --indices $indices --weights $weights \
    --init $init --max_iterations $max_iterations --restarts $restarts \
    --model $model --dataset $dataset --data_path $data_path \
    --num_images $num_images --num_exp $num_exp --target_id $target_id \
    --lr $lr --tv $tv \
    --generative_model $generative_model --gen_dataset $gen_dataset \
    --gias --gias_lr $gias_lr --gias_iterations $gias_iterations \
    --bn_stat $bn_stat --image_norm $image_norm --group_lazy $group_lazy \
    --checkpoint_path $checkpoint_path 

