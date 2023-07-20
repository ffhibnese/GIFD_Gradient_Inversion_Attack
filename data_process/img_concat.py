import PIL.Image as Image
import os

IMAGES_PATH = '/home/itml_fh/gradient-inversion-generative-image-prior/'  # 图片集地址
IMAGES_FORMAT = ['.jpg', '.JPG', '.png']  # 图片格式
IMAGE_SIZE = 64  # 每张小图片的大小
IMAGE_ROW = 1  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 6  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_DIR = '/home/itml_fh/gradient-inversion-generative-image-prior/data_process/concat_img/ffhq_main'  # 图片转换后的地址
IMAGE_NAME = 'ffhq_12000.png'
IMAGE_SAVE_PATH = os.path.join(IMAGE_SAVE_DIR, IMAGE_NAME)

# 获取图片集地址下的所有图片名称
# image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
#                os.path.splitext(name)[1] == item]
# image_names = ["outputs/gan_free/ffhq/ex1_1img_test/geiping/12000_gen.png", "outputs/gan_free/ffhq/ex1_1img_test/Yin/12000_gen.png", "outputs/ffhq/ex1_11imgs_cos/ggl/12000_gen.png", \
#  "outputs/ffhq/ex1_11imgs_cos/gias/12000_gen.png", "outputs/ffhq/ex1_11imgs_cos/Best_output/12000_gen.png","outputs/gan_free/ffhq/ex1_1img_test/12000_gt.png"]
# image_names = ["outputs/gan_free/ffhq/ex4_3img/geiping/1000_gen.png", "outputs/gan_free/ffhq/ex4_3img/Yin/1000_gen.png", "outputs/ffhq/ex1_10imgs_project/ggl/1000_gen.png", \
#  "/home/itml_fh/gradient-inversion-generative-image-prior/outputs/GIAS/ex1_4imgs(inclu gilo)/gias/1000_gen.png", "former_results/stylegan2_seed/ex1_manual_seed/Best_output/1000_gen.png","outputs/stylegan2_project/ex1/1000_gt.png"]
# image_names = ["outputs/GIAS/imagenet/ex3_50imgs_gias_real/11000_gt.png", "outputs/gan_free/imagenet/ex3_50imgs_64/geiping/11000_gen.png", "outputs/gan_free/imagenet/ex3_50imgs_64/Yin/11000_gen.png", "outputs/GGL/imagenet/ex1_50imgs_ggl_real/ggl/11000_gen.png", \
#  "outputs/GIAS/imagenet/ex3_50imgs_gias_real/gias/11000_gen.png", "outputs/ILO/imagenet/ex3_50imgs_ilo_new_r/Best_first_4_layer/11000_gen.png"]

# #For ffhq main

image_names = ["outputs/ILO/ffhq/ex3_70img_ffhq_IR_valid/12000_gt.png", "outputs/gan_free/ffhq/ex6_70img/Yin/12000_gen.png", "outputs/gan_free/ffhq/ex6_70img/geiping/12000_gen.png",  \
 "outputs/GGL/ffhq/ex1_70imgs/ggl/12000_gen.png","outputs/GIAS/ffhq/ex3_GIAS_70imgs/gias/12000_gen.png", "outputs/ILO/ffhq/ex3_70img_ffhq_IR_valid/Best_first_4_layer/12000_gen.png"]

#For FFHQ main

# image_names = ["outputs/ILO/ffhq/ex3_140img_ffhq_IR_valid/12000_gt.png", "outputs/gan_free/ffhq/ex14_140img/Yin/12000_gen.png", "outputs/gan_free/ffhq/ex14_140img/geiping/12000_gen.png",  \
#  "outputs/GGL/ffhq/ex1_140imgs/ggl/12000_gen.png","outputs/GIAS/ffhq/ex3_GIAS_140imgs/gias/12000_gen.png", "outputs/ILO/ffhq/ex3_140img_ffhq_IR_valid/Best_first_4_layer/12000_gen.png"]

#For ffhq ablation
# image_names = ["outputs/ILO/ffhq/ex3_50imgs_ilo_new_r/12000_gt.png" ,"outputs/ILO/ffhq/ex3_50imgs_ilo_new_r/Best_first_0_layer/12000_gen.png", "outputs/ILO/ffhq/ex5_50imgs_ilo_no_r/layer9/12000_gen.png",
#   "outputs/ILO/ffhq/ex5_50imgs_ilo_no_r/Best_first_9_layer/12000_gen.png", "outputs/ILO/ffhq/ex3_50imgs_ilo_new_r/Best_first_9_layer/12000_gen.png"]

#For ffhq ablation
# image_names = ["outputs/ILO/ffhq/ex3_140img_ffhq_IR_valid/0_gt.png" ,"outputs/ILO/ffhq/ex1_140img_ffhq_no_l1/layer0/0_gen.png", "outputs/ILO/ffhq/ex1_140img_ffhq_no_l1/layer4/0_gen.png",
#   "outputs/ILO/ffhq/ex1_140img_ffhq_no_l1/Best_first_4_layer/0_gen.png", "outputs/ILO/ffhq/ex3_140img_ffhq_IR_valid/Best_first_4_layer/0_gen.png"]

# #For ffhq ood

# image_names = ["ood_outputs/cartoon/imagenet/GILO/ex1_15img_car_ilo/7_gt.png", "ood_outputs/cartoon/imagenet/gan_free/ex1_15img_car_gan_free/Yin/7_gen.png", "ood_outputs/cartoon/imagenet/gan_free/ex1_15img_car_gan_free/geiping/7_gen.png",  \
#  "ood_outputs/cartoon/imagenet/GGL/ex1_15img_car_ggl/ggl/7_gen.png","ood_outputs/cartoon/imagenet/GIAS/ex1_15img_car_gias/gias/7_gen.png", "ood_outputs/cartoon/imagenet/GILO/ex1_15img_car_ilo/Best_first_4_layer/7_gen.png"]


# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) > IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("Too many images!")

image_gap = 3
# 定义图像拼接函数
def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE + image_gap * (IMAGE_COLUMN - 1), IMAGE_ROW * IMAGE_SIZE + image_gap * (IMAGE_ROW - 1)), (255, 255, 255))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            if IMAGE_COLUMN * (y - 1) + x - 1 >= len(image_names):
                break
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1])
            # from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
            #     (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * (IMAGE_SIZE + image_gap) , (y - 1) * (IMAGE_SIZE + image_gap)))
    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图


if not os.path.isdir(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)
image_compose()  # 调用函数
