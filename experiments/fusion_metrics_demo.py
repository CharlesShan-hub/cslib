import clib.metrics.fusion as metrics
print(metrics.vis.shape, metrics.fused.shape)
for (k,v) in metrics.info_summary_dict.items():
    print(f"{k}(CDDFuse)  : {v['metric'](metrics.ir,metrics.vis,metrics.cddfuse)}")
    print(f"{k}(DenseFuse): {v['metric'](metrics.ir,metrics.vis,metrics.densefuse)}")
    print(f"{k}(ADF)      : {v['metric'](metrics.ir,metrics.vis,metrics.adf)}")