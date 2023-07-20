from sre_constants import NOT_LITERAL
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import json
import os 

def draw_chart(x, y, x_label, y_label, title, save_path, multi_curve=False):
    print(x.to_list())
    x_str = [str(i) for i in x]
    if multi_curve:
        plt.figure()
        plt.xticks(range(len(x)), x_str)
        plt.xlabel(x_label)
        for _y, _y_label in zip(y, y_label):
            # plt.plot(x, _y, label = _y_label)
            plt.plot(range(len(x)), _y, label = _y_label)
        plt.legend()
    else:
        plt.figure()
        plt.xticks(range(len(x)), x_str)
        plt.xlabel(x_label)
        plt.ylabel(y_label) 
        plt.title(title)
        plt.plot(range(len(x)), y)
    plt.savefig(save_path)



# columns = ['idx', 'Start_layer', 'labels', 'ng_method', 'defense', 'd_param', 'trained_model', 'seed', 'n_trials', 'budget', 'adaptive', 'loss', 'MSE', 'LPIPS(VGG)', 'LPIPS(ALEX)', 'PSNR', 'RMSE', 'FMSE', 'TV-ori', 'TV-rec']  


with open('./data_process/config.json','r',encoding='utf8') as fp:
    json_data = json.load(fp)

input_dir = json_data['input_dir']
exp_name = json_data['exp_name']
n_trials = json_data["trial_num"]
output_dir = json_data["output_dir"]

dir_path = os.path.join(input_dir, exp_name)
for idx in range(n_trials):
    data = pd.DataFrame(columns=columns)
    for dir_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, dir_name, "log.csv")
        df = pd.read_csv(file_path)
        data = pd.concat([data, df], axis = 0)
    

# print(data)

    #draw the loss curve
    _x = data['Start_layer']
    _loss = data['loss']
    # print(_x)
    # print(_loss)
    x_label = "Start layer"
    y_label = "Loss"
    title = "Loss Curve"
    save_path = os.path.join(output_dir, exp_name, "Rec_Loss", "rec_loss.png")

    draw_chart(_x, _loss, x_label, y_label, title, save_path)

    #draw the LPIPS
    _y = [data["LPIPS(VGG)"].tolist(), data["LPIPS(ALEX)"].tolist()]
    y_label = ["LPIPS(VGG)", "LPIPS(ALEX)"]
    title = "LPIPS Curve"
    save_path = os.path.join(output_dir, exp_name, "LPIPS", "lpips.png")
    # save_path =  "./Charts/LPIPS_" + exp_name + ".png"

    multi_curve = True
    # print(_y[0])
    draw_chart(_x, _y, x_label, y_label, title, save_path, multi_curve)