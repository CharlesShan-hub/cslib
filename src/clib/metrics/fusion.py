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
# from metrics.q_s import *      # MEFB - 利用 SSIM 的指标 Piella's Fusion Quality Index
# from metrics.q_w import *      # MEFB - 利用 SSIM 的指标 Weighted Fusion Quality Index
# from metrics.q_e import *      # MEFB - 利用 SSIM 的指标 Piella's Edge-dependent Fusion Quality Index
# from metrics.q_c import *      # MEFB - Cvejic
# from metrics.q_y import *      # MEFB - Yang
# from metrics.mb import *       # Mean bias
from .collection.mae import *             # RS   - Mean absolute error
from .collection.mse import *             # VIFB - Mean squared error 均方误差
from .collection.rmse import *            # VIFB - Root mean squared error 均方误差
# from metrics.nrmse import *    # Normalized Root Mean Square Error
# from metrics.ergas import *    # Normalized Global Error
# from metrics.q_h import *      # OB
# from metrics.q import *        # REV
# from metrics.wfqi import *
# from metrics.efqi import *

# 图片信息
from .collection.ag import *              # VIFB - 平均梯度
from .collection.mg import *              # MA   - Mean Graident (similar to AG)
from .collection.ei import *              # VIFB - 边缘强度
# # from metrics.pfe import *      # Many
from .collection.sd import *              # VIFB - 标准差
from .collection.sf import *              # VIFB - 空间频率
from .collection.q_sf import *            # OE - 基于空间频率的指标 (metricZheng)
from .collection.q_abf import *           # VIFB - 基于梯度的融合性能
# from metrics.eva import *      # Zhihu - 点锐度
# from metrics.asm import *      # Zhihu - 角二阶矩 - 不可微!!!
# from metrics.sam import *      # Zhihu - 光谱角测度 - 要修改
from .collection.con import *      # 对比度
# from metrics.fmi import *      # OE - fmi_w(Discrete Meyer wavelet),fmi_g(Gradient),fmi_d(DCT),fmi_e(Edge),fmi_p(Raw pixels (no feature extraction))
# from metrics.q_p import *      # MEFB
# from metrics.n_abf import *    # Tang - 基于噪声评估的融合性能
# from metrics.pww import *      # Many - Pei-Wei Wang's algorithms

# 视觉感知
from .collection.q_cv import *            # VIFB - H. Chen and P. K. Varshney
from .collection.q_cb import *            # VIFB - 图像模糊与融合的质量评估 包含 cbb,cbm,cmd
from .collection.vif import *    # 视觉保真度 - 不可微!! 优化用 VIFF 就行
# from metrics.viff import *     # Tang - 视觉保真度

# 新指标暂时没分类
# from metrics.df import *       #
# from metrics.q_mi import *     #
# from metrics.q_s import *      #
# from metrics.q_e import *      #
# from metrics.uqi import *      # Many
# from metrics.qi import *       # Many
# from metrics.theta import *    # Many
# from metrics.fqi import *      # Many
# from metrics.fsm import *      # Many
# from metrics.wfqi import *     # Many
# from metrics.efqi import *     # Many
# from metrics.d import *        # Many

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
