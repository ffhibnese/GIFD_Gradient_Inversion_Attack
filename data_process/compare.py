import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    #ood
    gias_path = "/home/itml_fh/gradient-inversion-generative-image-prior/ood_outputs/cartoon/ffhq/GIAS/ex1_15img_car_gias/table_Metrics.csv"
    ggl_path = "/home/itml_fh/gradient-inversion-generative-image-prior/ood_outputs/cartoon/ffhq/GGL/ex1_15img_car_ggl/table_Metrics.csv"
    ilo_path = "/home/itml_fh/gradient-inversion-generative-image-prior/ood_outputs/cartoon/ffhq/GILO/ex1_15img_car_gilo/table_Metrics.csv"
    gan_free_path = "/home/itml_fh/gradient-inversion-generative-image-prior/ood_outputs/cartoon/ffhq/gan_free/ex1_15img_car_gan_free/table_Metrics.csv"
    
    #ffhq defense
    # gias_path = "/home/itml_fh/gradient-inversion-generative-image-prior/defense_outputs/additive_noise/ffhq/GIAS/ex1_30imgs_noise_gias/table_Metrics.csv"
    # ggl_path = "/home/itml_fh/gradient-inversion-generative-image-prior/defense_outputs/additive_noise/ffhq/GGL/ex1_30imgs_noise_ggl/table_Metrics.csv"
    # ilo_path = "/home/itml_fh/gradient-inversion-generative-image-prior/defense_outputs/additive_noise/ffhq/GIFD/ex1_30imgs_noise_gifd_new/table_Metrics.csv"
    # gan_free_path = "/home/itml_fh/gradient-inversion-generative-image-prior/defense_outputs/additive_noise/ffhq/gan_free/ex1_30imgs_noise_gan_free/table_Metrics.csv"

    #imagenet
    # gias_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/GIAS/imagenet/ex3_GIAS_70imgs/table_Metrics.csv"
    # ggl_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/GGL/imagenet/ex1_70imgs/table_Metrics.csv"
    # ilo_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/improve/imagenet/ex4_70img_imagenet/table_Metrics.csv"
    # gan_free_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/gan_free/imagenet/ex6_70img/table_Metrics.csv"

    # ffhq
    # gias_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/GIAS/ffhq/ex3_GIAS_70imgs/table_Metrics.csv"
    # ggl_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/GGL/ffhq/ex1_70imgs/table_Metrics.csv"
    # ilo_path = "outputs/ILO/ffhq/ex3_70img_ffhq_IR_valid/table_Metrics.csv"
    # gan_free_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/gan_free/ffhq/ex6_70img/table_Metrics.csv"


    #imagenet defense
    # gias_path = "/home/itml_fh/gradient-inversion-generative-image-prior/defense_outputs/gradient_sparsification/imagenet/GIAS/ex1_30imgs_sparse_imagenet_gias/table_Metrics.csv"
    # ggl_path = "/home/itml_fh/gradient-inversion-generative-image-prior/defense_outputs/gradient_sparsification/imagenet/GGL/ex1_30imgs_sparse_imagenet_ggl/table_Metrics.csv"
    # ilo_path = "/home/itml_fh/gradient-inversion-generative-image-prior/defense_outputs/gradient_sparsification/imagenet/GIFD/ex1_30imgs_sparse_imagenet_gifd/table_Metrics.csv"
    # gan_free_path = "/home/itml_fh/gradient-inversion-generative-image-prior/defense_outputs/gradient_sparsification/imagenet/gan_free/ex1_30imgs_sparse_gan_free/table_Metrics.csv"

    #ffhq defense
    # gias_path = "/home/itml_fh/gradient-inversion-generative-image-prior/defense_outputs/gradient_sparsification/ffhq/GIAS/ex1_30imgs_sparse_gias/table_Metrics.csv"
    # ggl_path = "/home/itml_fh/gradient-inversion-generative-image-prior/defense_outputs/gradient_sparsification/ffhq/GGL/ex1_30imgs_sparse_ggl/table_Metrics.csv"
    # ilo_path = "/home/itml_fh/gradient-inversion-generative-image-prior/defense_outputs/gradient_sparsification/ffhq/GIFD/ex1_30imgs_sparse_gifd/table_Metrics.csv"
    # gan_free_path = "/home/itml_fh/gradient-inversion-generative-image-prior/defense_outputs/gradient_sparsification/ffhq/gan_free/ex1_30imgs_sparse_gan_free/table_Metrics.csv"

    # gias_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/GIAS/imagenet/ex3_50imgs_gias_real/table_Metrics.csv"
    # ggl_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/GGL/imagenet/ex1_50imgs_ggl_real/table_Metrics.csv"
    # ilo_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/ILO/imagenet/ex3_50imgs_ilo_new_r/table_Metrics.csv"
    # gan_free_path = "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/gan_free/imagenet/ex3_50imgs_64/table_Metrics.csv"


    df_gias = pd.read_csv(gias_path, index_col=0) if gias_path else None
    df_ggl = pd.read_csv(ggl_path, index_col=0) if ggl_path else None
    df_ilo = pd.read_csv(ilo_path, index_col=0) if ilo_path else None
    df_gan_free = pd.read_csv(gan_free_path, index_col=0) if gan_free_path else None

    col_index = [i if i.startswith("layer") else '' for i in list(df_ilo.columns)]
    
    select_best_idx = [True if col.endswith('psnr') else False for col in col_index]
    chosen_layers = [layer[:-5] for layer in df_ilo.loc[:,select_best_idx].idxmin(1).tolist()]
    # print(chosen_layers)
    df_total = pd.DataFrame(index=['Yin', 'Geiping', 'GGL', 'GIAS', 'GILO'])
    for indicator in ['psnr', 'lpips(alex)', 'ssim', 'mse_i']:
        # df_stats = pd.concat([df_gias['gias_'+indicator], df_ggl['ggl_'+indicator], df_ilo['Best_output_'+indicator], \
        #  df_gan_free['Yin_'+indicator], df_gan_free['geiping_'+indicator]], axis=1, ignore_index=False)
        real_best_idx = [True if col.endswith(indicator) else False for col in col_index]

        chosen_cols = [col+'_'+indicator for col in chosen_layers]
        chosen_best = pd.Series([df_ilo.loc[row, col] for row, col in zip(df_ilo.index, chosen_cols)], index=df_ilo.index)
        
        # print(len(real_best_idx))
        # print(len(df_ilo.columns))
        # if indicator in ['psnr', 'ssim']:
        #     real_best = df_ilo.loc[:,real_best_idx].max(axis=1)
        # else:
        #     real_best = df_ilo.loc[:,real_best_idx].min(axis=1)

        df_stats = pd.concat([df_gan_free['Yin_'+indicator], df_gan_free['geiping_'+indicator], df_ggl['ggl_'+indicator], \
            df_gias['gias_'+indicator], df_ilo['Best_first_4_layer_'+indicator]], axis=1, ignore_index=False)
        
        df_total[indicator+"_mean"] = df_stats.mean(axis=0).tolist()
        df_total[indicator+"_std"] = df_stats.std(axis=0).tolist()

        # df_stats.mean(axis=0).to_csv("/home/itml_fh/gradient-inversion-generative-image-prior/data_process/csv_file/" + indicator + ".csv", header=0)
        print(df_stats.mean(axis=0))
        # print(df_stats)

    # out_dir = "/home/itml_fh/gradient-inversion-generative-image-prior/data_process/results/compare/ffhq"
    # out_dir = "/home/itml_fh/gradient-inversion-generative-image-prior/data_process/results/compare/defense/gradient_sparsification/ffhq/"
    out_dir = "/home/itml_fh/gradient-inversion-generative-image-prior/data_process/results/compare/ood_outputs/cartoon/ffhq/"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    df_total.to_csv(os.path.join(out_dir, "statistic_cmpr.csv"))