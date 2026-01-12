# base configurations
model = dict(
    type='GUMixRSSegmentation',
    clip_type='laion2b_s32b_b79k',
    model_type='ViT-H-14',
    dino_type='dinov3_sat',     # dinov3_sat, dinov3_vitl16_sat, dinov3_vitl16
    mask_generator='sam2',   # mask2former, sam2, entityseg, eomt, None
    multi_scales=[1.0, 1.5],
    use_georsclip=True,  # GeoRSCLIP
)
# ('metaclip_fullcc', 'ViT-B-16-quickgelu')
# ('metaclip_fullcc', 'ViT-L-14-quickgelu')
# ('laion2b_s32b_b79k', 'ViT-H-14')
# clip_type = 'openai',  # Recommended: use OpenAI CLIP weights as the base for ViT-B/32
# model_type = 'ViT-B-32',  # Switch to ViT-B/32 (a.k.a. ViT-B-32 / ViT-B/32)


test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=1.0, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=5))
