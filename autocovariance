import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
    
import torch.utils.data as data
import os.path as osp

#1ch＆1batch用の自己共分散関数
# def calculate_autocovariance(image, u, v):

#     # 画像の平均値を計算
#     mean = image.mean()

#     #分散を計算
#     variance = ((image - mean) ** 2).mean()
    
#     # 平均値を引く処理
#     image = image - mean
    
#     _,_, H, W = image.size()
    
#     # 資料2ページより外周パディング＆シフト操作
#     image1 = F.pad(image, (u, 0, v, 0), "constant", 0) #fs(x,y)
#     image2 = F.pad(image, (0, u, 0, v), "constant", 0) #fs(x + u,y + v)
    
#     # 確認用画像を生成
#     torchvision.utils.save_image(image1, "image1.png") #fs(x,y)
#     torchvision.utils.save_image(image2, "image2.png") #fs(x + u,y + v)

#     # 自己共分散関数Cを計算
#     autocovariance = (image1 * image2).mean()
    
#     # 自己相関係数を計算
#     print(autocovariance/variance)
#     return autocovariance/variance
def calculate_autocovariance(image, u, v):
    # 画像の平均値を計算
    mean = image.mean()

    # 分散を計算
    variance = ((image - mean) ** 2).mean()
    if variance == 0:
        raise ValueError("Variance is zero, cannot normalize.")

    # 平均値を引く処理
    image = image - mean

    _, _, H, W = image.size()

    # 画像をシフトさせて対応する部分を切り出し
    shifted_image1 = image[:, :, max(0, v):H, max(0, u):W]
    shifted_image2 = image[:, :, 0:H - max(0, v), 0:W - max(0, u)]

    # 確認用画像を生成
    # torchvision.utils.save_image(shifted_image1, "shifted_image1.png")  # fs(x, y)
    # torchvision.utils.save_image(shifted_image2, "shifted_image2.png")  # fs(x + u, y + v)

    # 自己共分散関数Cを計算
    autocovariance = (shifted_image1 * shifted_image2).mean()

    # 自己相関係数を計算
    autocorrelation = autocovariance / variance
    print(autocorrelation)

    return autocorrelation
    return autocorrelation

def make_datapath_list_for_Kodak(rootpath):
    # Create template for paths to image and annotation files
    imgpath_template = osp.join(rootpath, 'PNGImages', '%s.png')
    annopath_template = osp.join(rootpath, 'MaskImages', '%s.png')

    # Get ID (file name) of each file, training and validation
    val_id_names = osp.join(rootpath + 'ImageSets/mask.txt')

    # Create a list of paths to image files and annotation files for validation data
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # Strip blank spaces and line breaks
        img_path = (imgpath_template % file_id)  # Path of the image
        anno_path = (annopath_template % file_id)  # Path of annotation
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return val_img_list, val_anno_list

def prepare_dataset_Kodak(batch_size=1,rootpath = "./Kodak"):
    val_img_list, val_anno_list = make_datapath_list_for_Kodak(rootpath=rootpath)
    val_dataset = KodakDataset(val_img_list, val_anno_list, phase="test")

    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)

    return val_dataloader,val_img_list

class KodakDataset(data.Dataset):

    def __init__(self, img_list, anno_list, phase="test"):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)
        masked_image = torch.where((anno_class_img > 0), img, anno_class_img)
        maskdata = anno_class_img[0:1,:,:]
       
        images_with_alpha = torch.cat([masked_image, maskdata], dim=0)

        return masked_image, maskdata, img, anno_class_img, images_with_alpha
       

    def pull_item(self, index):
        # 1. Load Image
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   # [Height][Width][channnel]
        img = torchvision.transforms.functional.to_tensor(img)
        # 2. Load annotation
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)  
        anno_class_img = anno_class_img.convert("L").convert("RGB")
        anno_class_img = torchvision.transforms.functional.to_tensor(anno_class_img)
        return img, anno_class_img


def get_image_by_index(dataloader, index):
    # データセットから特定のインデックスのデータを取得
    masked_image, maskdata, img, anno_class_img, images_with_alpha = dataloader.dataset.__getitem__(index)
    masked_image, maskdata, img, anno_class_img, images_with_alpha = masked_image.unsqueeze(0), maskdata.unsqueeze(0), img.unsqueeze(0), anno_class_img.unsqueeze(0), images_with_alpha.unsqueeze(0)
    return masked_image, maskdata, img, anno_class_img, images_with_alpha

from statistics import mean
if __name__ == "__main__":
    rootpath = "./Kodak/"
    batch_size = 1  # 必要に応じてバッチサイズを調整してください

    # データセットとデータローダーの準備
    val_dataloader, val_img_list = prepare_dataset_Kodak(batch_size=batch_size, rootpath=rootpath)
    
    maskvals = []
    Ravgs = []
    Gavgs = []
    Bavgs = []

    for i in range(24):
        masked_image, maskdata, img, anno_class_img, images_with_alpha = get_image_by_index(val_dataloader, i)
        
        maskval = calculate_autocovariance(maskdata[:,0:1,:,:],1,1)
        # Ravg = calculate_autocovariance(img[:,0:1,:,:],1,1)
        # torchvision.utils.save_image(img[:,0:1,:,:],"Rimg.png")
        # Gavg = calculate_autocovariance(img[:,1:2,:,:],1,0)
        # Bavg = calculate_autocovariance(img[:,2:3,:,:],1,0)
        Ravg = calculate_autocovariance(masked_image[:,0:1,:,:],1,1)
        # # torchvision.utils.save_image(masked_image[:,0:1,:,:],"maskedRimg.png")
        Gavg = calculate_autocovariance(masked_image[:,1:2,:,:],1,1)
        Bavg = calculate_autocovariance(masked_image[:,2:3,:,:],1,1)
        maskvals.append(maskval.item())
        Ravgs.append(Ravg.item())
        Gavgs.append(Gavg.item())
        Bavgs.append(Bavg.item())

    print(f"R(ra)平均値: {mean(Ravgs):.3f}")
    print(f"G(ga)平均値: {mean(Gavgs):.3f}")
    print(f"B(ba)平均値: {mean(Bavgs):.3f}")
    print(f"RGB平均値: {(mean(Ravgs) + mean(Gavgs) + mean(Bavgs)) / 3:.3f}")
    print(f"Mask平均値: {mean(maskvals):.3f}")
