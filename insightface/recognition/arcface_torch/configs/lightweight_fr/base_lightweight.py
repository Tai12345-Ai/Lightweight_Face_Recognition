from easydict import EasyDict as edict

config = edict()

# =====================
# Backbone
# =====================
config.network = "mbf"  # "mbf" | "shufflefacenet" | "vargfacenet"
config.embedding_size = 512

# =====================
# Loss / Margin
# =====================
config.loss_type = "combined_margin"  # "combined_margin" | "adaface" | "magface"
config.margin_list = (1.0, 0.5, 0.0)  # (m1, m2, m3) — ArcFace default: m1=1, m2=0.5, m3=0
config.interclass_filtering_threshold = 0

# =====================
# Training
# =====================
config.resume = False
config.save_all_states = False
config.output = None  # auto-generated if None

config.fp16 = True
config.batch_size = 64  # smaller default for limited hardware
config.gradient_acc = 1

config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4

config.num_epoch = 40
config.warmup_epoch = 2

# Partial FC
config.sample_rate = 1.0

# =====================
# Data
# =====================
config.rec = "/path/to/dataset"  # path to MXNet RecordIO dataset
config.num_classes = 93431       # MS1MV3 default
config.num_image = 5179510       # MS1MV3 default
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
config.dali = False
config.dali_aug = False
config.num_workers = 2

# =====================
# Logging
# =====================
config.verbose = 2000
config.frequent = 10
config.seed = 2048

# WandB (disabled by default for lightweight project)
config.using_wandb = False
config.wandb_key = ""
config.suffix_run_name = None
config.wandb_entity = "entity"
config.wandb_project = "project"
config.wandb_log_all = True
config.save_artifacts = False
config.wandb_resume = False

# =====================
# Extensions (Phase 6-7, disabled by default)
# =====================
config.use_aroface = False
config.use_crfiqa = False
config.use_petalface = False
