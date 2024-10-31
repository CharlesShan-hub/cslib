from clib.dataset import Flowers17
from clib.algorithms.selective_search import selective_search
from clib.utils import path_to_rgb, to_image

import numpy as np
import cv2

class Flowers2(Flowers17):
    def __init__(self,
        root: str,
        is_svm: bool,
        image_size: int,
        threshold: float = 0.5,
        transform = None
        ):
        super().__init__(root)
        
        self.names = {
            0: 'Tulip',       # 561
            1: 'Pansy',       # 1281
            2: 'Background',  # for RCNN
        }
        self.d = {
            561:[90,126,350,434],
            562:[160,15,340,415],
            563:[42,25,408,405],
            564:[66,48,384,318],
            565:[17,208,363,354],
            566:[42,20,398,310],
            567:[40,60,410,290],
            568:[40,60,360,380],
            573:[140,80,360,360],
            574:[50,80,360,360],
            575:[140,80,400,350],
            576:[140,80,400,350],
            577:[100,200,400,200],
            578:[200,100,380,160],
            579:[20,180,520,180],
            580:[20,10,420,450],
            581:[60,100,400,300],
            582:[152,35,398,435],
            583:[40,45,380,315],
            584:[40,45,410,395],
            608:[200,40,200,250],
            610:[180,105,300,300],
            613:[180,105,150,150],
            617:[70,120,210,280],
            627:[90,80,230,150],
            630:[270,140,165,108],
            633:[140,200,120,120],
            634:[220,180,220,150],
            636:[220,88,200,232],
            637:[290,42,110,158],
            1281:[90,66,330,374],
            1282:[90,35,330,425],
            1283:[90,45,316,435],
            1284:[26,16,408,574],
            1285:[50,55,375,420],
            1286:[40,63,347,487],
            1287:[60,80,340,360],
            1288:[34,27,403,535],
            1289:[40,60,360,380],
            1290:[36,5,399,525],
            1291:[40,105,298,371],
            1292:[40,60,362,460],
            1293:[40,20,394,512],
            1294:[84,30,329,470],
            1295:[93,92,330,420],
            1296:[64,250,379,466],
            1297:[106,40,277,392],
            1298:[72,60,348,540],
            1305:[50,17,350,543],
            1306:[103,100,308,379],
            1307:[27,28,403,572],
            1314:[30,14,420,516],
            1315:[38,80,332,426],
            1316:[72,32,354,568],
            1317:[86,70,264,421],
            1318:[95,110,285,430],
            1319:[111,50,273,405],
            1323:[16,122,465,502],
            1324:[40,60,380,440],
            1325:[114,37,242,356],
        }
        self.is_svm = is_svm
        self.image_size = image_size
        self.threshold = threshold
        self._patch_folder = self._base_folder / 'patches'
        self.transform = transform
        self.images = []
        self.labels = []
        if self._patch_folder.exists():
            self.load_from_npy()
        else:
            self.save_to_numpy()

    def save_to_numpy(self):
        self._patch_folder.mkdir()

        for (key, box) in self.d.items():
            images = []
            labels = []
            img = path_to_rgb(self._images_folder / f"image_{key:04d}.jpg")
            _, regions = selective_search(img, scale=500, sigma=0.9, min_size=10)
            candidates = set()
            for r in regions:
                # excluding same rectangle (with different segments)
                if r['rect'] in candidates:
                    continue
                # excluding small regions
                if r['size'] < 220:
                    continue
                if (r['rect'][2] * r['rect'][3]) < 500:
                    continue
                # resize to 227 * 227 for input
                proposal_img, proposal_vertice = self.clip_pic(img, r['rect'])
                # Delete Empty array
                if len(proposal_img) == 0:
                    continue
                # Ignore things contain 0 or not C contiguous array
                x, y, w, h = r['rect']
                if w == 0 or h == 0:
                    continue
                # Check if any 0-dimension exist
                [a, b, c] = proposal_img.shape
                if a == 0 or b == 0 or c == 0:
                    continue
                resized_proposal_img = self.resize_image(proposal_img, self.image_size, self.image_size)
                candidates.add(r['rect'])
                img_float = np.asarray(resized_proposal_img, dtype="float32")
                images.append(img_float)
                # IOU
                iou_val = self.IOU(box, proposal_vertice)
                # labels, let 0 represent default class, which is background
                index = 1 if key > 1280 else 2 # 1 and 2 is two kind of flowers
                if self.is_svm:
                    if iou_val < self.threshold:
                        labels.append(0)
                    else:
                        labels.append(index)
                else:
                    label = np.zeros(2 + 1) # 2 flowers, one background
                    if iou_val < self.threshold:
                        label[0] = 1
                    else:
                        label[index] = 1
                    labels.append(label)
            np.save(self._patch_folder / f'image_{key}.npy', [images])
            np.save(self._patch_folder / f'label_{key}.npy', [labels])
            self.images.extend(images)
            self.labels.extend(labels)
    
    def clip_pic(self, img, rect):
        [x, y, w, h] = rect
        x_1 = x + w
        y_1 = y + h
        # return img[x:x_1, y:y_1, :], [x, y, x_1, y_1, w, h]   
        return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]
    
    def resize_image(self,in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
        img = cv2.resize(in_image, (new_width, new_height), resize_mode)
        if out_image:
            cv2.imwrite(out_image, img)
        return img
    
    def if_intersection(self,xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
        if_intersect = False
        if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
            if_intersect = True
        elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
            if_intersect = True
        elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
            if_intersect = True
        elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
            if_intersect = True
        else:
            return if_intersect
        if if_intersect:
            x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
            y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
            x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
            y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
            area_inter = x_intersect_w * y_intersect_h
            return area_inter
    
    def IOU(self, ver1, vertice2):
        # vertices in four points
        vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
        area_inter = self.if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
        if area_inter:
            area_1 = ver1[2] * ver1[3]
            area_2 = vertice2[4] * vertice2[5]
            iou = float(area_inter) / (area_1 + area_2 - area_inter)
            return iou
        return False
    
    def load_from_npy(self):
        for key in self.d:
            i = np.load(self._patch_folder / f'image_{key}.npy')[0]
            l = np.load(self._patch_folder / f'label_{key}.npy')[0]
            for n in range(i.shape[0]):
                self.images.append(i[n,:,:,:])
                self.labels.append(l[n,:])   
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image, label = to_image(self.images[idx]), np.argmax(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label