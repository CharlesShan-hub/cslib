import clib.metrics.fusion as metrics

for (k,v) in metrics.info_summary_dict.items():
    print(f"{k}: {v['metric'](metrics.ir,metrics.vis,metrics.fused)}")