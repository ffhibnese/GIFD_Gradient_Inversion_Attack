export CUDA_VISIBLE_DEVICES=6

data_path='/home/jjw/.torch/FFHQ'
# dataset="FFHQ64"
dataset="FFHQ"
# dataset="CIFAR10"
model="ResNet18"
init='randn'
cost_fn='sim'
indices='def'
weights='equal'


lr=3e-2
gias_lr=1e-3
meta_lr=1e-2


max_iterations=1000
gias_iterations=500
num_exp=200
num_images=4
restarts=1

# Priors
tv=1e-4
z_norm=1e-3
bn_stat=0
image_norm=0
group_lazy=0

# generative_model='stylegan2'
# gen_dataset='I512'
generative_model='stylegan2-ada-untrained'
# gen_dataset='FFHQ64'
gen_dataset='FFHQ'

checkpoint_path='results/G_b5229fd4ad112bdb836889d856209ec5_20.pkl'

target_id=92
# data_path='/home/jjw/.torch/ILSVRC2012'


# data_path='~/.torch/'


python -u train_gen.py \
    --unsigned --save_image \
    --cost_fn $cost_fn --indices $indices --weights $weights \
    --init $init --max_iterations $max_iterations --restarts $restarts \
    --model $model --dataset $dataset --data_path $data_path \
    --num_images $num_images --num_exp $num_exp \
    --lr $lr --tv $tv --meta_lr $meta_lr \
    --z_norm $z_norm \
    --generative_model $generative_model --gen_dataset $gen_dataset \
    --giml --gias_lr $gias_lr --gias_iterations $gias_iterations \
    --bn_stat $bn_stat --image_norm $image_norm --group_lazy $group_lazy 

