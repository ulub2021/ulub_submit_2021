from torch.utils.data.dataset import Dataset
import cv2, json, os
from pathlib import Path
from PIL import Image

def get_age_class(age):
    if age < 20:
        age_class = 0
    elif age >= 20 and age < 25:
        age_class = 1
    elif age >= 25 and age < 30:
        age_class = 2
    elif age >= 40 and age < 45:
        age_class = 3
    elif age >= 45 and age < 50:
        age_class = 4
    elif age >= 50 and age < 55:
        age_class = 5
    elif age >= 55 and age < 60:
        age_class = 6
    elif age >= 60 and age < 65:
        age_class = 7
    elif age >= 65 and age < 70:
        age_class = 8
    elif age >= 70:
        age_class = 9
    return age_class

def get_skintone_class(skintone):
    if skintone == 0:
        skintone_class = 0
    else:
        skintone_class = 1
    return skintone_class

class UTKFaceDataset(Dataset):
    def __init__(self, data_dir, json_dir, var=0.0, split=None, cls_type=None, transforms=None):
        """
        data_dir: directory path where data exist
        json_dir: directory path where data list json exist
        var: proportion of mixed split
        cls_type: 'pred'_'bias' - gender_skintone or skintone_gender
        split: ub1, ub2, test
        """

        data_path = Path(data_dir)
        self.data = []

        if split == 'test':
            with open(os.path.join(json_dir, 'skintone_gender_test.json'), 'r') as file:
                img_list = json.load(file)
        else:
            with open(os.path.join(json_dir, 'white_female.json'), 'r') as file:
                wf = json.load(file)
            with open(os.path.join(json_dir, 'white_male.json'), 'r') as file:
                wm = json.load(file)
            with open(os.path.join(json_dir, 'black_female.json'), 'r') as file:
                bf = json.load(file)
            with open(os.path.join(json_dir, 'black_male.json'), 'r') as file:
                bm = json.load(file)
            with open(os.path.join(json_dir, 'asian_female.json'), 'r') as file:
                af = json.load(file)
            with open(os.path.join(json_dir, 'asian_male.json'), 'r') as file:
                am = json.load(file)
            with open(os.path.join(json_dir, 'indian_female.json'), 'r') as file:
                indf = json.load(file)
            with open(os.path.join(json_dir, 'indian_male.json'), 'r') as file:
                indm = json.load(file)

            wf_len, wm_len, bf_len, bm_len, af_len, am_len, indf_len, indm_len \
                = len(wf), len(wm), len(bf), len(bm), len(af), len(am), len(indf), len(indm)
            wf_back, wf_data = wf[:int(var * wm_len)], wf[int(var * wm_len):]
            wm_back, wm_data = wm[:int(var * wf_len)], wm[int(var * wf_len):]
            bf_back, bf_data = bf[:int(var * bm_len)], bf[int(var * bm_len):]
            bm_back, bm_data = bm[:int(var * bf_len)], bm[int(var * bf_len):]
            af_back, af_data = af[:int(var * am_len)], af[int(var * am_len):]
            am_back, am_data = am[:int(var * af_len)], am[int(var * af_len):]
            indf_back, indf_data = indf[:int(var * indm_len)], indf[int(var * indm_len):]
            indm_back, indm_data = indm[:int(var * indf_len)], indm[int(var * indf_len):]


            if split == 'ub1':
                img_list = wf_data + wm_back[:int(len(wf_data) * var)] + \
                            bm_data + bf_back[:int(len(bm_data) * var)] + \
                            am_data + af_back[:int(len(am_data) * var)] + \
                            indm_data + indf_back[:int(len(indm_data) * var)]
            elif split == 'ub2':
                img_list = wm_data + wf_back[:int(len(wm_data) * var)] + \
                            bf_data + bm_back[:int(len(bf_data) * var)] + \
                            af_data + am_back[:int(len(af_data) * var)] + \
                            indf_data + indm_back[:int(len(indf_data) * var)]

        for img in img_list:
            if cls_type.split('_')[0] == 'skintone':
                self.data.append((data_path / img, get_skintone_class(int(img.split('_')[2]))))
            elif cls_type.split('_')[0] == 'gender':
                self.data.append((data_path / img, int(img.split('_')[1])))
            elif cls_type.split('_')[0] == 'age':
                self.data.append((data_path / img, get_age_class(int(img.split('_')[0]))))

        self.len = len(self.data)
        self.transforms = transforms

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = cv2.imread(str(img_path))
        try:
            img = Image.fromarray(img)
        except:
            print(img_path)
            input('Image does not exist.')
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return self.len