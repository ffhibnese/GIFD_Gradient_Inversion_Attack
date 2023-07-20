export CUDA_VISIBLE_DEVICES=1,2,3

i=1
# comp_rate=( 0 0.3 0.5 0.8 0.9 0.95 0.98 )
comp_rate=( 0 )
# T_ID=( 11000 13000 15000 17000 19000 )
T_ID=( 1000 )

dataset=( "I32" "I64" "I128" )
indices=( 'def' 'def' 'top50' )
tv=( 1e-6 1e-6 1e-6 )
max_iterations=( 500 7 1000 )
gias_iterations=( 5000 6 8000 )
lr=( 3e-2 3e-2 3e-2 )
gias_lr=( 1e-3 1e-3 1e-3 )

model="ResNet18"
init='randn'     #the initial distribution of latent code
# cost_fn='l2'
weights='equal'
# weights='linear'
# weights='exp'


num_exp=1
num_images=4
restarts=3

# Priors
tv=1e-6 # above
bn_stat=0
image_norm=0
group_lazy=0


generative_model='stylegan2'
gen_dataset='I512'
# gen_dataset='CIFAR10'

data_path='/home/jjw/.torch/ILSVRC2012'
# data_path='/home/kjc/.torch/FFHQ'
# data_path='/home/kjc/.torch/'



table_path=( 'i32_tables_ours' 'i64_tables_ours' 'i128_tables_ours' )
result_path=( 'i32_results_ours' 'i64_results_ours' 'i128_results_ours' )

for compression_rate in "${comp_rate[@]}"; do
    # $cost_fn="compressed$compression_rate" 
    for target_id in "${T_ID[@]}"; do
    python rec_mult.py --optim ours \
        --unsigned --save_image \
        --cost_fn "sim_cmpr$compression_rate"  --indices ${indices[i]} --weights $weights \
        --init $init --max_iterations ${max_iterations[i]} --gias_iterations ${gias_iterations[i]} --restarts $restarts \
        --model $model --dataset ${dataset[i]} --data_path $data_path \
        --num_images $num_images --num_exp $num_exp --target_id $target_id \
        --lr ${lr[i]} --tv ${tv[i]} --bn_stat $bn_stat --image_norm $image_norm --group_lazy $group_lazy \
        --generative_model $generative_model --gen_dataset $gen_dataset --gias --gias_lr ${gias_lr[i]} \
        --result_path ${result_path[i]} --table_path ${table_path[i]}
    done    
done


# --cost_fn "compressed$compression_rate"
# --cost_fn "sim_cmpr$compression_rate"
# dataset="I32"
# dataset="I64"
# dataset="I128"
# indices='top10' # For 128
# indices='top50' # For 64
# indices='def' # For 32
# tv=1e-8
# tv=1e-7
# tv=1e-6