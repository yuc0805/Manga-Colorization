_domain_a = None
_domain_b = None
model = dict(
    type='CycleGAN',
    generator=dict(
        type='ResnetGenerator',
        in_channels=3,
        out_channels=3,
        base_channels=64,
        norm_cfg=dict(type='IN'),
        use_dropout=False,
        num_blocks=9,
        padding_mode='reflect',
        init_cfg=dict(type='normal', gain=0.02)),
    discriminator=dict(
        type='PatchDiscriminator',
        in_channels=3,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='IN'),
        init_cfg=dict(type='normal', gain=0.02)),
    gan_loss=dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0),
    default_domain='colored',
    reachable_domains=['grayscale', 'colored'],
    related_domains=['grayscale', 'colored'],
    gen_auxiliary_loss=[
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(pred='cycle_grayscale', target='real_grayscale'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(pred='cycle_colored', target='real_colored'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(pred='identity_grayscale', target='real_grayscale'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(pred='identity_colored', target='real_colored'),
            reduction='mean')
    ])
train_cfg = dict(buffer_size=50)
test_cfg = None
train_dataset_type = 'UnpairedImageDataset'
val_dataset_type = 'UnpairedImageDataset'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
domain_a = 'grayscale'
domain_b = 'colored'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img_grayscale',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img_colored',
        flag='color'),
    dict(
        type='Resize',
        keys=['img_grayscale', 'img_colored'],
        scale=(256, 368),
        interpolation='bicubic'),
    dict(
        type='Crop',
        keys=['img_grayscale', 'img_colored'],
        crop_size=(368, 256),
        random_crop=True),
    dict(type='Flip', keys=['img_grayscale'], direction='horizontal'),
    dict(type='Flip', keys=['img_colored'], direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=['img_grayscale', 'img_colored']),
    dict(
        type='Normalize',
        keys=['img_grayscale', 'img_colored'],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=['img_grayscale', 'img_colored']),
    dict(
        type='Collect',
        keys=['img_grayscale', 'img_colored'],
        meta_keys=['img_grayscale_path', 'img_colored_path'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img_grayscale',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img_colored',
        flag='color'),
    dict(
        type='Resize',
        keys=['img_grayscale', 'img_colored'],
        scale=(256, 368),
        interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=['img_grayscale', 'img_colored']),
    dict(
        type='Normalize',
        keys=['img_grayscale', 'img_colored'],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=['img_grayscale', 'img_colored']),
    dict(
        type='Collect',
        keys=['img_grayscale', 'img_colored'],
        meta_keys=['img_grayscale_path', 'img_colored_path'])
]
data_root = None
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type='UnpairedImageDataset',
        dataroot='.././train_test/',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_grayscale',
                flag='color'),
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_colored',
                flag='color'),
            dict(
                type='Resize',
                keys=['img_grayscale', 'img_colored'],
                scale=(256, 368),
                interpolation='bicubic'),
            dict(
                type='Crop',
                keys=['img_grayscale', 'img_colored'],
                crop_size=(368, 256),
                random_crop=True),
            dict(type='Flip', keys=['img_grayscale'], direction='horizontal'),
            dict(type='Flip', keys=['img_colored'], direction='horizontal'),
            dict(
                type='RescaleToZeroOne', keys=['img_grayscale',
                                               'img_colored']),
            dict(
                type='Normalize',
                keys=['img_grayscale', 'img_colored'],
                to_rgb=False,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
            dict(type='ImageToTensor', keys=['img_grayscale', 'img_colored']),
            dict(
                type='Collect',
                keys=['img_grayscale', 'img_colored'],
                meta_keys=['img_grayscale_path', 'img_colored_path'])
        ],
        test_mode=False,
        domain_a='grayscale',
        domain_b='colored'),
    val=dict(
        type='UnpairedImageDataset',
        dataroot='.././train_test/',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_grayscale',
                flag='color'),
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_colored',
                flag='color'),
            dict(
                type='Resize',
                keys=['img_grayscale', 'img_colored'],
                scale=(256, 368),
                interpolation='bicubic'),
            dict(
                type='RescaleToZeroOne', keys=['img_grayscale',
                                               'img_colored']),
            dict(
                type='Normalize',
                keys=['img_grayscale', 'img_colored'],
                to_rgb=False,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
            dict(type='ImageToTensor', keys=['img_grayscale', 'img_colored']),
            dict(
                type='Collect',
                keys=['img_grayscale', 'img_colored'],
                meta_keys=['img_grayscale_path', 'img_colored_path'])
        ],
        test_mode=True,
        domain_a='grayscale',
        domain_b='colored'),
    test=dict(
        type='UnpairedImageDataset',
        dataroot='.././train_test/',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_grayscale',
                flag='color'),
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='img_colored',
                flag='color'),
            dict(
                type='Resize',
                keys=['img_grayscale', 'img_colored'],
                scale=(256, 368),
                interpolation='bicubic'),
            dict(
                type='RescaleToZeroOne', keys=['img_grayscale',
                                               'img_colored']),
            dict(
                type='Normalize',
                keys=['img_grayscale', 'img_colored'],
                to_rgb=False,
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
            dict(type='ImageToTensor', keys=['img_grayscale', 'img_colored']),
            dict(
                type='Collect',
                keys=['img_grayscale', 'img_colored'],
                meta_keys=['img_grayscale_path', 'img_colored_path'])
        ],
        test_mode=True,
        domain_a='grayscale',
        domain_b='colored'))
checkpoint_config = dict(interval=5000, by_epoch=False, save_optimizer=True)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=['fake_grayscale', 'fake_colored'],
        interval=5000)
]
runner = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '../cyclegan_lsgan_resnet_in_1x1_246200_summer2winter_convert-bgr_20210902_165932-fcf08dc1.pth'
resume_from = './work_dirs/cyclegan_bw2cl/ckpt/cyclegan_bw2cl/iter_100000.pth'
workflow = [('train', 1)]
find_unused_parameters = True
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
dataroot = '.././train_test/'
optimizer = dict(
    generators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
    discriminators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))
lr_config = dict(
    policy='Linear', by_epoch=False, target_lr=0, start=125000, interval=1250)
use_ddp_wrapper = True
total_iters = 250000
exp_name = 'cyclegan_bw2cl'
work_dir = './work_dirs/cyclegan_bw2cl'
num_images = 716
metrics = dict(
    FID=dict(type='FID', num_images=716, image_shape=(3, 368, 256)),
    IS=dict(
        type='IS',
        num_images=716,
        image_shape=(3, 368, 256),
        inception_args=dict(type='pytorch')))
evaluation = dict(
    type='TranslationEvalHook',
    target_domain='colored',
    interval=5000,
    metrics=[
        dict(type='FID', num_images=716, bgr2rgb=True),
        dict(type='IS', num_images=716, inception_args=dict(type='pytorch'))
    ],
    best_metric=['fid', 'is'])
gpu_ids = [0]
