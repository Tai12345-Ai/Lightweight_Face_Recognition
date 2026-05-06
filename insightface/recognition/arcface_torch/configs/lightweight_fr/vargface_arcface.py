from easydict import EasyDict as edict

config = edict()
config.network = "vargfacenet"
config.loss_type = "combined_margin"
config.margin_list = (1.0, 0.5, 0.0)
config.fp16 = True
config.batch_size = 128
config.lr = 0.1
config.weight_decay = 5e-4
config.num_epoch = 24
config.warmup_epoch = 1

# CASIA-WebFace dataset
config.rec = "/content/faces_webface_112x112"
config.num_classes = 10572
config.num_image = 490623
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
config.output = "work_dirs/casia_vargface_arcface"
config.save_all_states = True
