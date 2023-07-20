import pandas as pd
import os
# data_path = "./solve_ilo_ood/ex3_OOD/table_Metrics.csv"
data_path = "outputs/GIAS/imagenet"
# data_path = "/home/itml_fh/gradient-inversion-generative-image-prior/ood_outputs/GIAS/ffhq"
exp_name = "ex3_50imgs_gias_real"
file_name = "table_Metrics.csv"

# save_path = "/home/itml_fh/gradient-inversion-generative-image-prior/data_process/results/ood_outputs/GIAS/ffhq"
save_path = "/home/itml_fh/gradient-inversion-generative-image-prior/data_process/results//GIAS/imagenet"


out_dir = os.path.join(save_path, exp_name)

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)


data = pd.read_csv(os.path.join(data_path, exp_name, file_name), index_col=0)
columns = list(data.columns)
# col_index = [True if (i.startswith('Best') and not i.endswith('num')) or i.startswith('gias') or i.startswith('ggl') else False for i in columns]
col_index = [True if i.startswith('gias') else False for i in columns]

new_df = data.loc[:,col_index].copy()
# print(new_df)
# new_df.fillna(0)
new_df.loc[new_df.shape[0]] = [new_df[col].mean() for col in list(new_df.columns)] 
new_df = new_df.rename(index={new_df.shape[0] - 1:'Avg'})

new_df.loc[new_df.shape[0]] = [new_df[col].std() for col in list(new_df.columns)] 
new_df = new_df.rename(index={new_df.shape[0] - 1:'Std'})

new_df.loc[new_df.shape[0]] = [new_df[col].max() for col in list(new_df.columns)] 
new_df = new_df.rename(index={new_df.shape[0] - 1:'Max'})

print(new_df)
new_df.to_csv(os.path.join(save_path, exp_name, "statistics_gias.csv"))
# df = pd.concat([pd.DataFrame(new_df.mean(axis=0)), pd.DataFrame(new_df.std(axis=0))
# df = pd.concat([new_df.mean(axis=0), new_df.std(axis=0)], axis=0)

# df.to_csv("statistics.csv")