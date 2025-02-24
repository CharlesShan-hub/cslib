''' Metrics for Image Fusion
Introduction:
    * xx(): 默认输入都是 0-1 的张量
    * xx_metric(): 默认输入都是 0-1 的张量, 但会调整调用方法()的输入，会与 Matlab 源码一致
    * xx_approach_loss()：默认输入都是 0-1 的张量，用于趋近测试

Reference:
    VIFB: X. Zhang, P. Ye and G. Xiao, "VIFB: A Visible and Infrared Image Fusion Benchmark," 
        2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 
        Seattle, WA, USA, 2020, pp. 468-478, doi: 10.1109/CVPRW50498.2020.00060.
    MEFB: Zhang X. Benchmarking and comparing multi-exposure image fusion algorithms[J]. 
        Information Fusion, 2021, 74: 111-131.
    OE: Zheng Liu, Erik Blasch, Zhiyun Xue, Jiying Zhao, Robert Laganiere, and Wei Wu. 
        2012. Objective Assessment of Multiresolution Image Fusion Algorithms for Context 
        Enhancement in Night Vision: A Comparative Study. IEEE Trans. Pattern Anal. Mach. 
        Intell. 34, 1 (January 2012), 94-109. https://doi.org/10.1109/TPAMI.2011.109
    RS(没有 matlab 代码): Yuhendra, et al. “Assessment of Pan-Sharpening Methods Applied to Image Fusion of 
        Remotely Sensed Multi-Band Data.” International Journal of Applied Earth Observation 
        and Geoinformation, Aug. 2012, pp. 165-75, https://doi.org/10.1016/j.jag.2012.01.013.
    MA: J. Ma, Y. Ma, C. Li, Infrared and visible image fusion methods 
        and applications: A survey, Inf. Fusion 45 (2019) 153-178.
    Many: P. Jagalingam, Arkal Vittal Hegde, A Review of Quality Metrics for Fused Image, 
        Aquatic Procedia, Volume 4, 2015, Pages 133-142, ISSN 2214-241X, https://doi.org/10.1016/j.aqpro.2015.02.019.
    Zhihu: 
        https://blog.csdn.net/qq_49729636/article/details/134502721
        https://zhuanlan.zhihu.com/p/136013000
    Rev:
        A New Edge and Pixel-Based Image Quality Assessment Metric for Colour and Depth Images
        https://github.com/SeyedMuhammadHosseinMousavi/A-New-Edge-and-Pixel-Based-Image-Quality-Assessment-Metric-for-Colour-and-Depth-Images
    Tang:
        https://github.com/Linfeng-Tang/Image-Fusion/tree/main/General%20Evaluation%20Metric
'''

# 信息论
from .collection.ce import *              # VIFB - 交叉熵
from .collection.en import *              # VIFB - 信息熵
from .collection.te import *              # MEFB - tsallis熵
from .collection.mi import *              # VIFB - 互信息
from .collection.nmi import *             # MEFB - 标准化互信息
from .collection.q_ncie import *          # MEFB - 非线性相关性
from .collection.psnr import *            # VIFB - 峰值信噪比
from .collection.cc import *              # Tang - 相关系数
from .collection.scc import *             # pytorch - 空间相关系数
from .collection.scd import *             # Tang - 差异相关和

# 结构相似性
from .collection.ssim import *            # VIFB - 结构相似度测量
from .collection.ms_ssim import *         # Tang - 多尺度结构相似度测量
from .collection.q_s import *             # OE   - 利用 SSIM 的指标 Piella's Fusion Quality Index
from .collection.q import *               # REV  - Image Quality Index (q, q0, qi, uqi, uiqi)
from .collection.q_w import *             # OE   - 利用 SSIM 的指标 Weighted Fusion Quality Index(wfqi)
from .collection.q_e import *             # OE   - 利用 SSIM 的指标 Piella's Edge-dependent Fusion Quality Index(efqi)
from .collection.q_c import *             # MEFB - Cvejic
from .collection.q_y import *             # MEFB - Yang
from .collection.mb import *              # Many - Mean bias (遥感的指标)
from .collection.mae import *             # RS   - Mean absolute error
from .collection.mse import *             # VIFB - Mean squared error 均方误差
from .collection.rmse import *            # VIFB - Root mean squared error 均方误差
from .collection.nrmse import *           # Normalized Root Mean Square Error
from .collection.ergas import *           # Zhihu- Normalized Global Error (遥感的指标)
from .collection.d import *               # Many - Degree of Distortion (遥感的指标)
# from .collection.q_h import *             # OB   - 每个图都要用小波，战略性放弃

# 图片信息
from .collection.ag import *              # VIFB - 平均梯度
from .collection.mg import *              # MA   - Mean Graident (similar to AG)
from .collection.ei import *              # VIFB - 边缘强度
from .collection.pfe import *             # Many - 百分比拟合误差 Percentage fit error
from .collection.sd import *              # VIFB - 标准差 sd / std / theta
from .collection.sf import *              # VIFB - 空间频率
from .collection.q_sf import *            # OE   - 基于空间频率的指标 (metricZheng)
from .collection.q_abf import *           # VIFB - 基于梯度的融合性能
from .collection.eva import *             # Zhihu- 点锐度 (遥感的指标, 中文期刊)
from .collection.asm import *             # Zhihu- 角二阶矩 - 不可微!!! (遥感的指标)
from .collection.sam import *             # Zhihu- 光谱角测度 - 要修改 (遥感的指标)
from .collection.con import *             # 对比度
from .collection.fmi import *             # OE   - fmi_w(Discrete Meyer wavelet),fmi_g(Gradient),fmi_d(DCT),fmi_e(Edge),fmi_p(Raw pixels (no feature extraction))
from .collection.n_abf import *           # Tang - 基于噪声评估的融合性能
from .collection.pww import *             # Many - Pei-Wei Wang's algorithms (matlab的跑不起来了,python的可以)
# from .collection.q_p import *             # MEFB 没翻译完

# 视觉感知
from .collection.q_cv import *            # VIFB - H. Chen and P. K. Varshney
from .collection.q_cb import *            # VIFB - 图像模糊与融合的质量评估 包含 cbb,cbm,cmd
from .collection.vif import *             # MEFB - 视觉保真度
from .collection.viff import *            # Tang - 视觉保真度 ( VIF for fusion)

# 新指标暂时没分类
# from .collection.df import *       #
# from .collection.q_mi import *     #
# from .collection.q_s import *      #
# from .collection.fqi import *      # Many
# from .collection.fsm import *      # Many

info_summary_dict = {
    'en':{
        'type': 'Information Theory',
        'name': 'Entropy',
        'zh': '信息熵',
        'metric':en_metric
    },
    'ce':{
        'type': 'Information Theory',
        'name': 'Cross Entropy',
        'zh': '交叉熵',
        'metric':ce_metric
    },
    'mi':{
        'type': 'Information Theory',
        'name': 'Mutual Information',
        'zh': '互信息',
        'metric':mi_metric
    },
    'psnr':{
        'type': 'Information Theory',
        'name': 'Peak Signal-to-Noise Ratio',
        'zh': '峰值信噪比',
        'metric':psnr_metric
    },
    'ssim':{
        'type': 'Structural Similarity',
        'name': 'Structural Similarity',
        'zh': '结构相似度',
        'metric':ssim_metric
    },
    'rmse':{
        'type': 'Structural Similarity',
        'name': 'Root Mean Square Error',
        'zh': '均方根误差',
        'metric':rmse_metric
    },
    'ag':{
        'type': 'Image Feature',
        'name': 'Average Gradient',
        'zh': '平均梯度',
        'metric':ag_metric
    },
    'ei':{
        'type': 'Image Feature',
        'name': 'Edge Intensity',
        'zh': '边缘强度',
        'metric':ei_metric
    },
    'sf':{
        'type': 'Image Feature',
        'name': 'Spatial Frequency',
        'zh': '空间频率',
        'metric':sf_metric
    },
    'q_abf':{
        'type': 'Image Feature',
        'name': 'Qabf',
        'zh': '基于梯度的融合性能',
        'metric':q_abf_metric
    },
    'sd':{
        'type': 'Image Feature',
        'name': 'Standard Deviation',
        'zh': '标准差',
        'metric':sd_metric
    },
    'q_cb':{
        'type': 'Visual Perception',
        'name': 'Metric of Chen Blum',
        'zh': 'Qcb',
        'metric':q_cb_metric
    },
    'q_cv':{
        'type': 'Visual Perception',
        'name': 'Metric of Chen',
        'zh': 'Qcv',
        'metric':q_cv_metric
    }
}

''' Demo: Use Example Images. (Run in experinment folder)
import clib.metrics.fusion as metrics
need = ['ag'] # Write metrics names that you need
for (k,v) in metrics.info_summary_dict.items():
    if k not in need: continue
    print(f"{k}(CDDFuse)  : {v['metric'](metrics.ir,metrics.vis,metrics.cddfuse)}")
    print(f"{k}(DenseFuse): {v['metric'](metrics.ir,metrics.vis,metrics.densefuse)}")
    print(f"{k}(ADF)      : {v['metric'](metrics.ir,metrics.vis,metrics.adf)}")
'''

from pathlib import Path
from ..utils import to_tensor,path_to_gray

__all__ = [
    'ir', 'vis', 'fused', 'cddfuse', 'densefuse', 'adf'
]

def load_demo_image():
    path = Path(__file__).resolve().parent / 'resources'
    filenames = ['ir', 'vis', 'CDDFuse', 'CDDFuse', 'DenseFuse', 'ADF']
    return [to_tensor(path_to_gray(path / f'{f}.png')).unsqueeze(0) for f in filenames]

(ir, vis, fused, cddfuse, densefuse, adf) = load_demo_image()
