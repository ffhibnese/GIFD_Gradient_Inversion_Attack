import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
#[200, 1000, 500, 300, 300, 200, 200, 200, 100, 100, 100, 50, 50]
def draw_bar(x, y, y_label, title, save_path):

    plt.figure()
    plt.xticks(range(len(x)), x)
    plt.xlabel('Layer')
    plt.ylabel(y_label) 
    plt.title(y_label.upper() + title)
    plt.bar(range(len(x)), y, width=0.8)
    plt.savefig(save_path)
# data_path = "./solve_ilo_ood/ex3_OOD/table_Metrics.csv"
data_path = "/home/itml_fh/gradient-inversion-generative-image-prior/defense_outputs/additive_noise/ffhq/GIFD"
# data_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/ILO/ffhq"
# data_path = "/home/itml_fh/gradient-inversion-generative-image-prior/ood_outputs/GILO/ffhq"


# data_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/ILO/ffhq"
# data_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/ILO/ffhq/ex1_ILO_70imgs/table_Metrics.csv"
# data_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/GIAS/ffhq/ex3_GIAS_70imgs/table_Metrics.csv"
# save_path = "/home/itml_fh/gradient-inversion-generative-image-prior/data_process/results/ILO/ffhq"
save_path = "/home/itml_fh/gradient-inversion-generative-image-prior/data_process/results/defense_outputs/additive_noise/ffhq/GIFD"
# save_path = "/home/itml_fh/gradient-inversion-generative-image-prior/data_process/results/ILO/ffhq"

# save_path = "/home/itml_fh/gradient-inversion-generative-image-prior/data_process/results/ood_outputs/GILO/ffhq"

# exp_name = "ex26_ILO_70imgs_8layer_nol1"
# exp_name = "ex25_ILO_70imgs_8layer"
# exp_name = "ex3_70img_ffhq_IR_valid"
exp_name = "ex1_30imgs_noise_gifd_new"

# exp_name = "ex6_50imgs_new_r_lastTry_valid"
# exp_name = "ex3_50imgs_new_r_again"
# exp_name = "ex3_50imgs_ilo_new_r_no_IR"
# exp_name = "ex3_50imgs_ilo_new_r_valid"
# exp_name = "ex6_50imgs_new_r_lastTry"


# exp_name = "ex2_30img_ffhq_no_r"

file_name = "table_Metrics.csv"

data = pd.read_csv(os.path.join(data_path, exp_name, file_name), index_col=0)
data = data.replace(-1, np.nan)
# print(data)
columns = list(data.columns)
col_index = [True if i.startswith('layer') or i.startswith('Best_output') or i.startswith('Best_f') else False for i in columns]
new_df = data.loc[:,col_index].copy()
# print(new_df)
# new_df.fillna(0)

new_df.loc['Avg'] = [new_df[col].mean(skipna=True) for col in list(new_df.columns)] 
# new_df = new_df.rename(index={new_df.shape[0] - 1:'Avg'})

new_df.loc['Std'] = [new_df[col].std(skipna=True) for col in list(new_df.columns)] 
# new_df = new_df.rename(index={new_df.shape[0] - 1:'Std'})

new_df.loc['Max'] = [new_df[col].max(skipna=True) for col in list(new_df.columns)] 
# new_df = new_df.rename(index={new_df.shape[0] - 1:'Max'})

layer_num = 8
out_dir = os.path.join(save_path, exp_name)

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

# for indicator in ['psnr', 'lpips(alex)', 'ssim', 'mse_i']:
#     # col_index = [f'layer{i}_' + indicator for i in range(layer_num)]
#     # col_index += ['Best_first_5layers_' + indicator, 'Best_output_' + indicator]
#     col_index = [f"Best_first_{i+1}_layer_" + indicator for i in range(layer_num)]
#     data_avg = new_df.loc['Avg', col_index]
#     draw_bar([f"{i}" for i in range(layer_num)] + ["Best_5", "Best"], data_avg, indicator, '_Avg', os.path.join(out_dir, f"{indicator}.png")  )
#     print(data_avg)

# print(new_df)
new_df.to_csv(os.path.join(save_path, exp_name, "statistics.csv"))

# df = pd.concat([pd.DataFrame(new_df.mean(axis=0)), pd.DataFrame(new_df.std(axis=0))
# df = pd.concat([new_df.mean(axis=0), new_df.std(axis=0)], axis=0)

# df.to_csv("statistics.csv")