'''
 * 方法(): 默认输入都是 0-1 的张量
 * 方法_metric(): 默认输入都是 0-1 的张量, 但会调整调用方法()的输入，会与 VIFB 一致
 * 方法_approach_loss()：默认输入都是 0-1 的张量，用于趋近测试
'''

# TODO
# EN, CE, MI有一些例子, 再改一改

# 打开一个注释需要满足的要求
# 1. 更改了 if __main__里边的测试函数
# 2. 写了 Reference
# 3. 下边的字典完善了
# 4. 找到了对应的 matlab 代码,并且修改完毕

# 信息论
from .ce import *              # VIFB - 交叉熵
from .en import *              # VIFB - 信息熵
# from metrics.te import *       # MEFB - tsallis熵
from .mi import *              # VIFB - 互信息
# from metrics.nmi import *      # MEFB - 标准化互信息
# from metrics.q_ncie import *   # MEFB - 非线性相关性
# from metrics.snr import *      # Many - 信噪比
from .psnr import *            # VIFB - 峰值信噪比
# from .cc import *       # Tang - 相关系数(正在改)
# from metrics.scc import *
# from metrics.scd import *      # Tang - 差异相关和

# 结构相似性
from .ssim import *            # VIFB - 结构相似度测量
# from metrics.ms_ssim import *  # Tang - 多尺度结构相似度测量
# from metrics.q_s import *      # MEFB - 利用 SSIM 的指标 Piella's Fusion Quality Index
# from metrics.q_w import *      # MEFB - 利用 SSIM 的指标 Weighted Fusion Quality Index
# from metrics.q_e import *      # MEFB - 利用 SSIM 的指标 Piella's Edge-dependent Fusion Quality Index
# from metrics.q_c import *      # MEFB - Cvejic
# from metrics.q_y import *      # MEFB - Yang
# from metrics.mb import *       # Mean bias
# from metrics.eme import *
# from metrics.mae import *
# from metrics.mse import *      # VIFB - 均方误差
from .rmse import *            # VIFB - 均方误差
# from metrics.nrmse import *    # Normalized Root Mean Square Error
# from metrics.ergas import *    # Normalized Global Error
# from metrics.q_h import *      # OB
# from metrics.q import *        # REV
# from metrics.wfqi import *
# from metrics.efqi import *

# 图片信息
from .ag import *              # VIFB - 平均梯度
from .ei import *              # VIFB - 边缘强度
# # from metrics.pfe import *      # Many
from .sd import *              # VIFB - 标准差
from .sf import *              # VIFB - 空间频率
# from metrics.q_sf import *     # OE - 基于空间频率的指标
from .q_abf import *           # VIFB - 基于梯度的融合性能
# from metrics.eva import *      # Zhihu - 点锐度
# from metrics.asm import *      # Zhihu - 角二阶矩 - 不可微!!!
# from metrics.sam import *      # Zhihu - 光谱角测度 - 要修改
# from metrics.con import *      # 对比度
# from metrics.fmi import *      # OE - fmi_w(Discrete Meyer wavelet),fmi_g(Gradient),fmi_d(DCT),fmi_e(Edge),fmi_p(Raw pixels (no feature extraction))
# from metrics.q_p import *      # MEFB
# from metrics.n_abf import *    # Tang - 基于噪声评估的融合性能
# from metrics.pww import *      # Many - Pei-Wei Wang's algorithms

# 视觉感知
from .q_cv import *            # VIFB - H. Chen and P. K. Varshney
from .q_cb import *            # VIFB - 图像模糊与融合的质量评估 包含 cbb,cbm,cmd
# from metrics.vif import *      # 视觉保真度 - 不可微!! 优化用 VIFF 就行
# from metrics.viff import *     # Tang - 视觉保真度

# 新指标暂时没分类
# # from metrics.df import *       #
# # from metrics.q_mi import *     #
# # from metrics.q_s import *      #
# # from metrics.q_e import *      #
# # from metrics.uqi import *      # Many
# # from metrics.qi import *       # Many
# # from metrics.theta import *    # Many
# # from metrics.fqi import *      # Many
# # from metrics.fsm import *      # Many
# # from metrics.wfqi import *     # Many
# # from metrics.efqi import *     # Many
# # from metrics.d import *        # Many


from .utils import *


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