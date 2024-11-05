from clib.utils import glance, path_to_rgb, to_tensor
from clib.algorithms.selective_search import selective_search
from model import AlexNet, RegNet
from utils import crop_and_filter_regions,show_rect

import numpy as np
import torch
import joblib
device = 'cpu'

INPUT_GT = [90,66,330,374]
INPUT_KEY = 1281
INPUT_PATH = f"/Users/kimshan/resources/DataSets/torchvision/flowers-17/jpg/image_{INPUT_KEY}.jpg"
FINETUNE_PATH = r"/Users/kimshan/resources/DataSets/Model/RCNN/Flowers17/AlexNet_Finetune/checkpoints/51.pt"
SVM1_PATH = r"/Users/kimshan/resources/DataSets/Model/RCNN/Flowers17/SVM/checkpoints/1_svm.pkl"
SVM2_PATH = r"/Users/kimshan/resources/DataSets/Model/RCNN/Flowers17/SVM/checkpoints/2_svm.pkl"
REG_PATH = r"/Users/kimshan/resources/DataSets/Model/RCNN/Flowers17/REG/checkpoints/21.pt"

def test():
    # Load Finetune Model
    finetune_model = AlexNet(
            num_classes=3,
            classify=False,
            save_feature=True
        )
    params = torch.load(
        FINETUNE_PATH, 
        map_location=device,
        weights_only=True
    )
    finetune_model.load_state_dict(params['model_state_dict'])

    # Load SVM Model
    svm1_model = joblib.load(SVM1_PATH)
    svm2_model = joblib.load(SVM2_PATH)

    # Load Reg Model
    reg_model = RegNet()
    params = torch.load(
        REG_PATH, 
        map_location=device,
        weights_only=True
    )
    reg_model.load_state_dict(params['model_state_dict'])


    # Get Input picture
    image = path_to_rgb(INPUT_PATH)
    # glance(image)

    # Selected Search
    _, regions = selective_search(image, scale=500, sigma=0.9, min_size=10)
    images,labels,rects = crop_and_filter_regions(image, INPUT_KEY, regions, INPUT_GT, 224, 0.5)

    # Get Features
    images_tensor = torch.stack([to_tensor(i) for i in images])
    features = finetune_model(images_tensor)

    # 
    results = []
    results_old = []
    results_label = []
    
    for i in range(len(images)):
        for svm in [svm1_model,svm2_model]:
            feature = features[i,:].data.cpu().numpy()
            pred = svm.predict([feature.tolist()])
            if pred[0] == 0:
                continue
            box = reg_model(features[i,:])
            if box[0]<0.3:
                continue
            # print(box)
            box = box.data.cpu().numpy()
            px, py, pw, ph = rects[i][0], rects[i][1], rects[i][2], rects[i][3]
            old_center_x, old_center_y = px + pw / 2.0, py + ph / 2.0
            x_ping, y_ping, w_suo, h_suo = box[1], box[2], box[3], box[4],
            new__center_x = x_ping * pw + old_center_x
            new__center_y = y_ping * ph + old_center_y
            new_w = pw * np.exp(w_suo)
            new_h = ph * np.exp(h_suo)
            new_verts = [new__center_x, new__center_y, new_w, new_h]
            results.append(new_verts)
            results_label.append(pred[0])

    flower = {1:'pancy', 2:'Tulip'}
    average_center_x, average_center_y, average_w,average_h = 0, 0, 0, 0
    #use average values to represent the final result
    for vert in results:
        average_center_x += vert[0]
        average_center_y += vert[1]
        average_w += vert[2]
        average_h += vert[3]
    average_center_x = average_center_x / len(results)
    average_center_y = average_center_y / len(results)
    average_w = average_w / len(results)
    average_h = average_h / len(results)
    average_result = [[average_center_x, average_center_y, average_w, average_h]]
    result_label = max(results_label, key=results_label.count)
    print(result_label,average_result)
    # show_rect(image, results_old, ' ')
    show_rect(image, average_result, flower[result_label])

if __name__ == "__main__":
    test()