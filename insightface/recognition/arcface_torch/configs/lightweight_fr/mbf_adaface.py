from easydict import EasyDict as edict

config = edict()
config.network = "mbf"
config.loss_type = "adaface"
config.fp16 = True
config.batch_size = 64
config.lr = 0.1
config.weight_decay = 1e-4
config.num_epoch = 40
config.warmup_epoch = 2

# AdaFace hyperparameters
config.adaface_s = 64.0
config.adaface_m = 0.4
config.adaface_h = 0.333
config.adaface_t_alpha = 0.01

# Dataset — update these paths for your setup
config.rec = "/path/to/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
