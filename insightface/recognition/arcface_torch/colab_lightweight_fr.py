# %% [markdown]
# # Lightweight Face Recognition Robust to Low-Quality Images
#
# **Project scope**: So sánh lightweight backbone ở **Representation stage** (Detection & Alignment giữ cố định).
#
# **Phase 1**: 3 backbone × ArcFace → clean + degraded eval → chọn 1-2 backbone tốt nhất.
# **Phase 2** (sau): Backbone tốt nhất × AdaFace / MagFace (không làm trong notebook này).
#
# **Runtime**: Google Colab T4 GPU.

# %% [markdown]
# ---
# ## Cell 1: Check GPU + Install Dependencies

# %%
# ============================================================
#  GLOBAL CONFIG — Chỉnh ở đây trước khi chạy
# ============================================================

RUN_MODE = "debug"        # "debug" (5 epochs) hoặc "report" (15 epochs)
BATCH_SIZE = 128           # Giảm xuống 64 hoặc 32 nếu OOM
RUN_TRAINING = True
RUN_EVAL = True
RUN_BENCHMARK = True

DEBUG_EPOCHS = 5
REPORT_EPOCHS = 15
NUM_EPOCH = DEBUG_EPOCHS if RUN_MODE == "debug" else REPORT_EPOCHS

print(f"RUN_MODE       = {RUN_MODE}")
print(f"NUM_EPOCH      = {NUM_EPOCH}")
print(f"BATCH_SIZE     = {BATCH_SIZE}")
print(f"RUN_TRAINING   = {RUN_TRAINING}")
print(f"RUN_EVAL       = {RUN_EVAL}")
print(f"RUN_BENCHMARK  = {RUN_BENCHMARK}")

# %%
# --- Check GPU ---
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"VRAM: {vram:.1f} GB")
else:
    raise RuntimeError("GPU NOT AVAILABLE! Runtime -> Change runtime type -> T4 GPU")

# %%
# --- Install dependencies ---
# numpy<1.24 for mxnet compatibility; mxnet==1.9.1 for reading .bin eval files
!pip install -q "numpy<1.24" easydict ptflops gdown opencv-python scikit-learn
!pip install -q mxnet==1.9.1

# NOTE: mxnet is ONLY needed to read .bin verification files (lfw.bin, cfp_fp.bin, agedb_30.bin).
# If mxnet fails to install on your Colab runtime, evaluation will need a different data format.
# In that case, see InsightFace docs for converting .bin to image folder + pairs.txt.

# Verify mxnet
try:
    import mxnet
    print(f"mxnet OK: {mxnet.__version__}")
except ImportError:
    print("WARNING: mxnet not installed. .bin evaluation files cannot be loaded.")

# %% [markdown]
# ---
# ## Cell 2: Clone InsightFace + Set Working Directory

# %%
import os

if not os.path.exists('/content/insightface'):
    !git clone --depth 1 https://github.com/deepinsight/insightface.git /content/insightface

WORK_DIR = '/content/insightface/recognition/arcface_torch'
os.chdir(WORK_DIR)
print(f"Working directory: {os.getcwd()}")

# %% [markdown]
# ---
# ## Cell 3: Create / Patch Backbone Files
#
# - **MobileFaceNet**: Already in repo (`backbones/mobilefacenet.py`).
# - **ShuffleFaceNet**: ShuffleNetV2-style face recognition backbone + GDC head. *(Not a faithful reproduction of any specific paper; inspired by ShuffleNetV2 architecture.)*
# - **VarGFaceNet**: Simplified VarGFaceNet-style compact backbone with SE modules + GDC head. *(Simplified version; does not implement the full variable-group convolution from the ICCVW 2019 paper.)*

# %%
# =============================================
#  ShuffleFaceNet (ShuffleNetV2-style backbone)
# =============================================
shufflefacenet_code = r'''"""
ShuffleFaceNet: ShuffleNetV2-style face recognition backbone.

NOTE: This is a ShuffleNetV2-style backbone adapted for face recognition,
NOT a faithful reproduction of any specific "ShuffleFaceNet" paper.
Uses channel shuffle blocks + GDC embedding head.
Input: 3x112x112  Output: 512-d embedding.
"""
import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    cpg = c // groups
    x = x.view(b, groups, cpg, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(b, -1, h, w)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        branch_features = oup // 2

        if stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.PReLU(branch_features),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(inp, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.PReLU(branch_features),
                nn.Conv2d(branch_features, branch_features, 3, stride, 1,
                          groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.PReLU(branch_features),
            )
        else:
            assert inp == branch_features * 2
            self.branch1 = None
            self.branch2 = nn.Sequential(
                nn.Conv2d(branch_features, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.PReLU(branch_features),
                nn.Conv2d(branch_features, branch_features, 3, 1, 1,
                          groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.PReLU(branch_features),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class ShuffleFaceNet(nn.Module):
    """ShuffleNetV2-style backbone for face recognition."""

    def __init__(self, fp16=False, num_features=512, width_mult=1.0, **kwargs):
        super().__init__()
        self.fp16 = fp16
        stage_repeats = [4, 8, 4]
        if width_mult == 1.0:
            stage_out_channels = [24, 116, 232, 464, 1024]
        elif width_mult == 0.5:
            stage_out_channels = [24, 48, 96, 192, 1024]
        elif width_mult == 1.5:
            stage_out_channels = [24, 176, 352, 704, 1024]
        else:
            raise ValueError(f"Unsupported width_mult: {width_mult}")

        inp = 3
        out0 = stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, out0, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out0), nn.PReLU(out0),
        )
        inp = out0
        self.stages = nn.ModuleList()
        for i, repeats in enumerate(stage_repeats):
            out = stage_out_channels[i + 1]
            layers = [InvertedResidual(inp, out, 2)]
            for _ in range(repeats - 1):
                layers.append(InvertedResidual(out, out, 1))
            self.stages.append(nn.Sequential(*layers))
            inp = out

        final_ch = stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(inp, final_ch, 1, bias=False),
            nn.BatchNorm2d(final_ch), nn.PReLU(final_ch),
        )
        self.gdc = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, 7, groups=final_ch, bias=False),
            nn.BatchNorm2d(final_ch),
        )
        self.linear = nn.Linear(final_ch, num_features, bias=False)
        self.bn = nn.BatchNorm1d(num_features)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=self.fp16):
            x = self.conv1(x)
            for stage in self.stages:
                x = stage(x)
            x = self.conv5(x)
        x = self.gdc(x.float() if self.fp16 else x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.bn(x)
        return x


def get_shufflefacenet(fp16=False, num_features=512, **kwargs):
    return ShuffleFaceNet(fp16=fp16, num_features=num_features)
'''

with open('backbones/shufflefacenet.py', 'w') as f:
    f.write(shufflefacenet_code)
print("[OK] backbones/shufflefacenet.py")

# %%
# =============================================
#  VarGFaceNet (Simplified VarGFaceNet-style)
# =============================================
vargfacenet_code = r'''"""
Simplified VarGFaceNet-style compact backbone for face recognition.

NOTE: This is a simplified version inspired by VarGFaceNet (ICCVW 2019).
It does NOT implement the full variable-group convolution from the original paper.
Uses depthwise-separable conv blocks with SE modules + GDC embedding head.
Input: 3x112x112  Output: 512-d embedding.
"""
import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ch, ch // reduction, 1, bias=False)
        self.act = nn.PReLU(ch // reduction)
        self.fc2 = nn.Conv2d(ch // reduction, ch, 1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        s = self.sig(self.fc2(self.act(self.fc1(self.pool(x)))))
        return x * s


class VarGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        self.use_se = use_se
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, stride, 1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch), nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if use_se:
            self.se = SEModule(out_ch)
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.layers(x)
        if self.use_se:
            out = self.se(out)
        if self.use_residual:
            out = out + x
        else:
            out = out + self.shortcut(x)
        return out


class VarGFaceNet(nn.Module):
    """Simplified VarGFaceNet-style compact backbone."""

    def __init__(self, fp16=False, num_features=512, **kwargs):
        super().__init__()
        self.fp16 = fp16
        channels = [40, 80, 160, 320]
        num_blocks = [3, 7, 4]
        self.head = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[0]), nn.PReLU(channels[0]),
            nn.Conv2d(channels[0], channels[0], 3, 1, 1, groups=channels[0], bias=False),
            nn.BatchNorm2d(channels[0]), nn.PReLU(channels[0]),
        )
        self.stages = nn.ModuleList()
        in_c = channels[0]
        for i in range(3):
            out_c = channels[i + 1]
            blocks = [VarGBlock(in_c, out_c, stride=2)]
            for _ in range(num_blocks[i] - 1):
                blocks.append(VarGBlock(out_c, out_c, stride=1))
            self.stages.append(nn.Sequential(*blocks))
            in_c = out_c
        self.embed_conv = nn.Sequential(
            nn.Conv2d(channels[-1], 512, 1, bias=False),
            nn.BatchNorm2d(512), nn.PReLU(512),
        )
        self.gdc = nn.Sequential(
            nn.Conv2d(512, 512, 7, groups=512, bias=False),
            nn.BatchNorm2d(512),
        )
        self.linear = nn.Linear(512, num_features, bias=False)
        self.bn = nn.BatchNorm1d(num_features)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=self.fp16):
            x = self.head(x)
            for stage in self.stages:
                x = stage(x)
        x = self.embed_conv(x.float() if self.fp16 else x)
        x = self.gdc(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.bn(x)
        return x


def get_vargfacenet(fp16=False, num_features=512, **kwargs):
    return VarGFaceNet(fp16=fp16, num_features=num_features)
'''

with open('backbones/vargfacenet.py', 'w') as f:
    f.write(vargfacenet_code)
print("[OK] backbones/vargfacenet.py")

# %%
# === Patch backbones/__init__.py ===
init_file = 'backbones/__init__.py'
with open(init_file, 'r') as f:
    content = f.read()

if 'shufflefacenet' not in content:
    patch = '''
    elif name == "shufflefacenet":
        from .shufflefacenet import get_shufflefacenet
        return get_shufflefacenet(fp16=kwargs.get("fp16", False),
                                  num_features=kwargs.get("num_features", 512))

    elif name == "vargfacenet":
        from .vargfacenet import get_vargfacenet
        return get_vargfacenet(fp16=kwargs.get("fp16", False),
                               num_features=kwargs.get("num_features", 512))

    else:
        raise ValueError()'''
    content = content.replace('    else:\n        raise ValueError()', patch)
    with open(init_file, 'w') as f:
        f.write(content)
    print("[OK] Patched backbones/__init__.py")
else:
    print("[OK] Already patched")

# Verify all 3 backbones
from backbones import get_model
x = torch.randn(2, 3, 112, 112).cuda()
for name in ['mbf', 'shufflefacenet', 'vargfacenet']:
    m = get_model(name, fp16=False, num_features=512).cuda().eval()
    with torch.no_grad():
        y = m(x)
    p = sum(pp.numel() for pp in m.parameters()) / 1e6
    print(f"  {name:<20} output={list(y.shape)}  params={p:.2f}M")
    del m
torch.cuda.empty_cache()
print("[OK] All backbones verified")

# %% [markdown]
# ---
# ## Cell 4: Create Configs

# %%
import os
os.makedirs('configs/lightweight_fr', exist_ok=True)

# --- Base config ---
base_config = f'''from easydict import EasyDict as edict

config = edict()

# Backbone
config.network = "mbf"
config.embedding_size = 512

# Loss
config.loss_type = "combined_margin"   # Phase 1: ArcFace only
config.margin_list = (1.0, 0.5, 0.0)  # (m1, m2, m3) ArcFace default
config.interclass_filtering_threshold = 0

# Training
config.resume = False
config.save_all_states = True
config.output = None
config.fp16 = True
config.batch_size = {BATCH_SIZE}
config.gradient_acc = 1
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4
config.num_epoch = {NUM_EPOCH}
config.warmup_epoch = 1
config.sample_rate = 1.0

# Data (CASIA-WebFace defaults)
config.rec = "/content/faces_webface_112x112"
config.num_classes = 10572
config.num_image = 490623
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
config.dali = False
config.dali_aug = False
config.num_workers = 4

# Logging
config.verbose = 2000
config.frequent = 50
config.seed = 2048

# WandB (disabled)
config.using_wandb = False
config.wandb_key = ""
config.suffix_run_name = None
config.wandb_entity = ""
config.wandb_project = ""
config.wandb_log_all = True
config.save_artifacts = False
config.wandb_resume = False
'''

with open('configs/lightweight_fr/__init__.py', 'w') as f:
    f.write('')
with open('configs/lightweight_fr/base_lightweight.py', 'w') as f:
    f.write(base_config)
print(f"[OK] base_lightweight.py (epochs={NUM_EPOCH}, batch={BATCH_SIZE})")

# --- Per-backbone configs ---
configs_spec = {
    'mbf_arcface':     ('mbf',            'casia_mbf_arcface'),
    'shuffle_arcface': ('shufflefacenet',  'casia_shuffle_arcface'),
    'vargface_arcface':('vargfacenet',     'casia_vargface_arcface'),
}

for fname, (net, outdir) in configs_spec.items():
    code = f'''from easydict import EasyDict as edict
config = edict()
config.network = "{net}"
config.output = "work_dirs/{outdir}"
'''
    with open(f'configs/lightweight_fr/{fname}.py', 'w') as f:
        f.write(code)
    print(f"[OK] configs/lightweight_fr/{fname}.py")

# %% [markdown]
# ---
# ## Cell 5: Create `train_lightweight.py`
#
# Single-GPU T4 compatible. Uses `torchrun --standalone --nproc_per_node=1`.
# Reuses `get_dataloader`, `CombinedMarginLoss`, `PartialFC_V2` from repo.

# %%
train_script = r'''#!/usr/bin/env python3
"""
train_lightweight.py — Single-GPU friendly training for lightweight FR.

Usage:
    torchrun --standalone --nproc_per_node=1 train_lightweight.py configs/lightweight_fr/mbf_arcface.py
"""
import argparse, importlib.util, logging, os, os.path as osp
import torch
from torch import distributed
from torch.utils.data import DataLoader

from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging


def load_config(config_file):
    """Load base_lightweight.py then overlay specific config."""
    base_path = osp.join(osp.dirname(osp.abspath(__file__)),
                         "configs", "lightweight_fr", "base_lightweight.py")
    spec = importlib.util.spec_from_file_location("base", base_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.config

    spec2 = importlib.util.spec_from_file_location("job", osp.abspath(config_file))
    mod2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod2)
    cfg.update(mod2.config)

    if cfg.output is None:
        cfg.output = osp.join("work_dirs", osp.splitext(osp.basename(config_file))[0])
    return cfg


def main(args):
    # --- Init distributed (single-GPU via torchrun --standalone) ---
    distributed.init_process_group(backend="nccl")
    rank = distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = distributed.get_world_size()

    cfg = load_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    torch.cuda.set_device(local_rank)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    # --- Data ---
    train_loader = get_dataloader(
        cfg.rec, local_rank, cfg.batch_size,
        cfg.dali, cfg.dali_aug, cfg.seed, cfg.num_workers)

    # --- Backbone ---
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16,
        num_features=cfg.embedding_size).cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(
        backbone, broadcast_buffers=False,
        device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.train()
    # NOTE: _set_static_graph() removed — can cause issues with custom backbones.

    # --- Loss + Head ---
    margin_loss = CombinedMarginLoss(
        64, cfg.margin_list[0], cfg.margin_list[1],
        cfg.margin_list[2], cfg.interclass_filtering_threshold)

    module_partial_fc = PartialFC_V2(
        margin_loss, cfg.embedding_size, cfg.num_classes,
        cfg.sample_rate, False)
    module_partial_fc.train().cuda()

    # --- Optimizer ---
    opt = torch.optim.SGD(
        [{"params": backbone.parameters()},
         {"params": module_partial_fc.parameters()}],
        lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    # --- LR scheduler ---
    total_batch = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // total_batch * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // total_batch * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt, warmup_iters=cfg.warmup_step, total_iters=cfg.total_step)

    # --- Resume ---
    start_epoch, global_step = 0, 0
    ckpt_path = os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt")
    if cfg.resume and os.path.exists(ckpt_path):
        d = torch.load(ckpt_path)
        start_epoch = d["epoch"]
        global_step = d["global_step"]
        backbone.module.load_state_dict(d["state_dict_backbone"])
        module_partial_fc.load_state_dict(d["state_dict_softmax_fc"])
        opt.load_state_dict(d["state_optimizer"])
        lr_scheduler.load_state_dict(d["state_lr_scheduler"])
        del d
        logging.info(f"Resumed from epoch {start_epoch}, step {global_step}")

    for k, v in cfg.items():
        logging.info(f": {k:<25} {v}")

    # --- Callbacks ---
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec,
        summary_writer=None, wandb_logger=None)
    callback_logging = CallBackLogging(
        frequent=cfg.frequent, total_step=cfg.total_step,
        batch_size=cfg.batch_size, start_step=global_step, writer=None)

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    # --- Training loop ---
    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss = module_partial_fc(local_embeddings, local_labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16,
                                 lr_scheduler.get_last_lr()[0], amp)
                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        # Save checkpoint every epoch
        if cfg.save_all_states:
            torch.save({
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
            }, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        if rank == 0:
            torch.save(backbone.module.state_dict(),
                       os.path.join(cfg.output, "model.pt"))
        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        final_path = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), final_path)
        logging.info(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    main(parser.parse_args())
'''

with open('train_lightweight.py', 'w') as f:
    f.write(train_script)
print("[OK] train_lightweight.py")

# %% [markdown]
# ---
# ## Cell 6: Create Degradation Module
#
# **Core degradations** (3 only):
# 1. `gaussian_blur` — Gaussian blur with variable sigma
# 2. `low_resolution` — Downsample then upsample back to 112x112
# 3. `low_illumination` — Gamma correction to darken image

# %%
os.makedirs('degradation', exist_ok=True)

with open('degradation/__init__.py', 'w') as f:
    f.write('from .transforms import DegradationTransform, SUPPORTED_DEGRADATIONS\n')

deg_code = r'''"""
Image degradation transforms for evaluation — core set only.

All transforms accept/return numpy array (H, W, 3) uint8.
Severity levels: 1 (mild) to 5 (severe).
"""
import cv2
import numpy as np

# Core degradations for this project
SUPPORTED_DEGRADATIONS = [
    "gaussian_blur",
    "low_resolution",
    "low_illumination",
]

_GAUSSIAN_BLUR_SIGMA = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.5, 5: 5.0}
_LOW_RES_SIZE = {1: 56, 2: 42, 3: 28, 4: 20, 5: 14}
_LOW_ILLUM_GAMMA = {1: 1.3, 2: 1.6, 3: 2.0, 4: 2.7, 5: 3.5}


def apply_gaussian_blur(image, severity=1):
    sigma = _GAUSSIAN_BLUR_SIGMA.get(severity, 2.0)
    ksize = int(np.ceil(sigma * 3)) * 2 + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_low_resolution(image, severity=1):
    h, w = image.shape[:2]
    low = _LOW_RES_SIZE.get(severity, 28)
    small = cv2.resize(image, (low, low), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_low_illumination(image, severity=1):
    gamma = _LOW_ILLUM_GAMMA.get(severity, 2.0)
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


_DEGRADATION_FN = {
    "gaussian_blur": apply_gaussian_blur,
    "low_resolution": apply_low_resolution,
    "low_illumination": apply_low_illumination,
}


class DegradationTransform:
    """Configurable, reproducible image degradation."""

    def __init__(self, degradation_type, severity=1, seed=42):
        if degradation_type not in SUPPORTED_DEGRADATIONS:
            raise ValueError(
                f"Unknown: {degradation_type}. Supported: {SUPPORTED_DEGRADATIONS}")
        if not 1 <= severity <= 5:
            raise ValueError(f"Severity must be 1-5, got {severity}")
        self.degradation_type = degradation_type
        self.severity = severity
        self._fn = _DEGRADATION_FN[degradation_type]

    def apply(self, image):
        return self._fn(image, severity=self.severity)

    def __repr__(self):
        return f"DegradationTransform({self.degradation_type}, s={self.severity})"
'''

with open('degradation/transforms.py', 'w') as f:
    f.write(deg_code)
print("[OK] degradation/ module (3 core degradations)")

# %% [markdown]
# ---
# ## Cell 7: Create `eval_degraded.py`

# %%
eval_code = r'''#!/usr/bin/env python3
"""
eval_degraded.py — Clean + degraded evaluation for face recognition.

Usage:
    python eval_degraded.py --network mbf --weight model.pt --rec /path/to/data
    python eval_degraded.py --network mbf --weight model.pt --rec /path/to/data \
        --degradations gaussian_blur,low_resolution,low_illumination --severities 1,3,5
"""
import argparse, os, pickle, sys
import numpy as np
import sklearn.preprocessing
import torch, torch.nn as nn

try:
    import mxnet as mx
except ImportError:
    mx = None

from backbones import get_model
from eval.verification import evaluate
from degradation.transforms import DegradationTransform, SUPPORTED_DEGRADATIONS


@torch.no_grad()
def load_bin_as_numpy(path, image_size=(112, 112)):
    assert mx is not None, "mxnet is required to load .bin verification files"
    try:
        with open(path, 'rb') as f: bins, issame_list = pickle.load(f)
    except UnicodeDecodeError:
        with open(path, 'rb') as f: bins, issame_list = pickle.load(f, encoding='bytes')
    n = len(issame_list) * 2
    images = np.empty((n, image_size[0], image_size[1], 3), dtype=np.uint8)
    for i in range(n):
        img = mx.image.imdecode(bins[i]).asnumpy()
        if img.shape[0] != image_size[0]:
            img = mx.image.resize_short(mx.nd.array(img), image_size[0]).asnumpy()
        images[i] = img
    return images, issame_list


@torch.no_grad()
def extract_embeddings(images, backbone, batch_size=64, device='cuda',
                       embedding_size=512):
    """Extract embeddings with flip augmentation (same as verification.test)."""
    n = images.shape[0]
    emb_list = []
    for flip in [False, True]:
        emb = np.zeros((n, embedding_size), dtype=np.float32)
        i = 0
        while i < n:
            j = min(i + batch_size, n)
            batch = images[i:j].copy()
            if flip:
                batch = batch[:, :, ::-1, :].copy()
            t = torch.from_numpy(batch.transpose(0, 3, 1, 2).astype(np.float32))
            t = ((t / 255.0) - 0.5) / 0.5
            emb[i:j] = backbone(t.to(device)).cpu().numpy()
            i = j
        emb_list.append(emb)
    return sklearn.preprocessing.normalize(emb_list[0] + emb_list[1])


def eval_condition(images, issame, backbone, degradation=None,
                   batch_size=64, device='cuda', embedding_size=512):
    imgs = images.copy()
    if degradation is not None:
        for i in range(len(imgs)):
            imgs[i] = degradation.apply(imgs[i])
    emb = extract_embeddings(imgs, backbone, batch_size, device, embedding_size)
    _, _, accuracy, val, val_std, far = evaluate(emb, issame, nrof_folds=10)
    return np.mean(accuracy), np.std(accuracy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='mbf')
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--rec', type=str, required=True)
    parser.add_argument('--targets', type=str, default='lfw,cfp_fp,agedb_30')
    parser.add_argument('--degradations', type=str, default='')
    parser.add_argument('--severities', type=str, default='1,3,5')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embedding_size', type=int, default=512)
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(',')]
    severities = [int(s) for s in args.severities.split(',')]
    degradations = ([d.strip() for d in args.degradations.split(',')]
                    if args.degradations else [])
    for d in degradations:
        if d not in SUPPORTED_DEGRADATIONS:
            print(f"ERROR: Unknown degradation '{d}'. Supported: {SUPPORTED_DEGRADATIONS}")
            sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = get_model(args.network, dropout=0, fp16=False,
                         num_features=args.embedding_size)
    backbone.load_state_dict(torch.load(args.weight, map_location=device))
    backbone = backbone.to(device).eval()

    results = {}
    for tgt in targets:
        bp = os.path.join(args.rec, tgt + '.bin')
        if not os.path.exists(bp):
            print(f'WARNING: {bp} not found, skipping'); continue
        print(f'\n=== {tgt.upper()} ===')
        images, issame = load_bin_as_numpy(bp)
        results[tgt] = {}

        acc, std = eval_condition(images, issame, backbone,
                                  batch_size=args.batch_size, device=device,
                                  embedding_size=args.embedding_size)
        results[tgt]['clean'] = (acc, std)
        print(f'  Clean: {acc*100:.2f}%')

        for dname in degradations:
            for sev in severities:
                deg = DegradationTransform(dname, severity=sev)
                a, s = eval_condition(images, issame, backbone, deg,
                                      args.batch_size, device, args.embedding_size)
                key = f'{dname}_s{sev}'
                results[tgt][key] = (a, s)
                drop = (acc - a) * 100
                print(f'  {key:<30} {a*100:.2f}% (drop: {drop:+.2f}%)')

    # Summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    for tgt, tr in results.items():
        ca = tr['clean'][0]
        print(f'\n--- {tgt.upper()} ---')
        print(f'  {"Condition":<30} {"Acc":>8} {"Drop":>8}')
        print(f'  {"-"*30} {"-"*8} {"-"*8}')
        for cond, (a, _) in tr.items():
            d = '---' if cond == 'clean' else f'{(ca-a)*100:+.2f}%'
            print(f'  {cond:<30} {a*100:>7.2f}% {d:>8}')

if __name__ == '__main__':
    main()
'''

with open('eval_degraded.py', 'w') as f:
    f.write(eval_code)
print("[OK] eval_degraded.py (embedding_size parameterized)")

# %% [markdown]
# ---
# ## Cell 8: Download + Verify CASIA-WebFace Dataset

# %%
import os, gdown

DATASET_DIR = "/content/faces_webface_112x112"
REQUIRED_FILES = ["train.rec", "train.idx", "lfw.bin", "cfp_fp.bin", "agedb_30.bin"]

if not os.path.exists(DATASET_DIR):
    # GDrive file ID from InsightFace _datasets_ README
    url = "https://drive.google.com/uc?id=1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l"
    zipfile = "/content/faces_webface_112x112.zip"

    print("Downloading CASIA-WebFace dataset (~260MB)...")
    print("If auto-download fails, download manually from:")
    print("  https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view")
    print("  OR Baidu Pan: https://pan.baidu.com/s/1AfHdPsxJZBD8kBJeIhmq1w")
    print("  Then upload .zip to /content/ and run: !unzip -q /content/<file>.zip -d /content/\n")

    try:
        gdown.download(url, zipfile, quiet=False)
        !unzip -q {zipfile} -d /content/
        !rm -f {zipfile}
    except Exception as e:
        print(f"\nAuto-download failed: {e}")
        print("Please download manually (see instructions above).")
        raise
else:
    print(f"Dataset already exists: {DATASET_DIR}")

# --- Verify required files ---
print("\nDataset verification:")
all_ok = True
for fname in REQUIRED_FILES:
    fpath = os.path.join(DATASET_DIR, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath) / 1024 / 1024
        print(f"  [OK] {fname:<20} {size:.1f} MB")
    else:
        print(f"  [MISSING] {fname}")
        all_ok = False

if not all_ok:
    raise FileNotFoundError(
        f"Dataset incomplete! Missing files in {DATASET_DIR}. "
        f"Required: {REQUIRED_FILES}. "
        f"Re-download or check the zip contents.")

print("\n[OK] Dataset verified — all required files present.")

# %% [markdown]
# ---
# ## Cell 9: Debug Run — Train MobileFaceNet first (sanity check)
#
# > **Purpose**: Verify the full pipeline works (data loading, loss, training, eval callbacks)
# > before committing GPU time to all 3 backbones.
# >
# > This is NOT for concluding which backbone is better.

# %%
if RUN_TRAINING:
    os.chdir(WORK_DIR)
    print(f"=== DEBUG RUN: MobileFaceNet + ArcFace ({NUM_EPOCH} epochs) ===")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Expected time: ~{NUM_EPOCH * 8} minutes on T4\n")
    !torchrun --standalone --nproc_per_node=1 train_lightweight.py configs/lightweight_fr/mbf_arcface.py
    print("\n[OK] MobileFaceNet debug run complete.")
    print("Check training log above for LFW/CFP-FP/AgeDB-30 accuracy.")
    print("If accuracy looks reasonable, proceed to train all 3 backbones in Cell 10.")
else:
    print("SKIPPED (RUN_TRAINING=False)")

# %% [markdown]
# ---
# ## Cell 10: Train All 3 Backbones with ArcFace
#
# > Only run this AFTER verifying Cell 9 completed successfully.
# >
# > **Fair comparison**: All 3 backbones trained from scratch, same dataset, same loss, same epochs.
# > Do NOT use pretrained weights for comparison — only for sanity check (Cell 9).

# %%
if RUN_TRAINING:
    os.chdir(WORK_DIR)

    # MobileFaceNet already trained in Cell 9 if same config.
    # Skip if model.pt already exists from debug run.
    mbf_model = "work_dirs/casia_mbf_arcface/model.pt"
    if os.path.exists(mbf_model):
        print(f"[SKIP] MobileFaceNet — already trained ({mbf_model})")
    else:
        print("=== Training MobileFaceNet ===")
        !torchrun --standalone --nproc_per_node=1 train_lightweight.py configs/lightweight_fr/mbf_arcface.py

    print("\n=== Training ShuffleFaceNet ===")
    !torchrun --standalone --nproc_per_node=1 train_lightweight.py configs/lightweight_fr/shuffle_arcface.py

    print("\n=== Training VarGFaceNet ===")
    !torchrun --standalone --nproc_per_node=1 train_lightweight.py configs/lightweight_fr/vargface_arcface.py

    print("\n[OK] All 3 backbones trained.")
else:
    print("SKIPPED (RUN_TRAINING=False)")

# %% [markdown]
# ---
# ## Cell 11: Clean Evaluation — Compare 3 Backbones

# %%
if RUN_EVAL:
    os.chdir(WORK_DIR)
    DATASET = "/content/faces_webface_112x112"

    print("=" * 70)
    print("  CLEAN EVALUATION: 3 Backbones × ArcFace")
    print("=" * 70)

    models = [
        ('mbf',            'work_dirs/casia_mbf_arcface/model.pt'),
        ('shufflefacenet', 'work_dirs/casia_shuffle_arcface/model.pt'),
        ('vargfacenet',    'work_dirs/casia_vargface_arcface/model.pt'),
    ]

    for net, wpath in models:
        if not os.path.exists(wpath):
            print(f"\n[SKIP] {net}: {wpath} not found")
            continue
        print(f"\n--- {net.upper()} ---")
        !python eval_degraded.py --network {net} --weight {wpath} --rec {DATASET}
else:
    print("SKIPPED (RUN_EVAL=False)")

# %% [markdown]
# ---
# ## Cell 12: Degraded Evaluation — 3 Core Degradations Only
#
# Degradations: `gaussian_blur`, `low_resolution`, `low_illumination`
# Severities: 1 (mild), 3 (moderate), 5 (severe)

# %%
if RUN_EVAL:
    os.chdir(WORK_DIR)
    DATASET = "/content/faces_webface_112x112"
    DEGRADATIONS = "gaussian_blur,low_resolution,low_illumination"
    SEVERITIES = "1,3,5"

    print("=" * 70)
    print("  DEGRADED EVALUATION: 3 Backbones × 3 Degradations × 3 Severities")
    print("=" * 70)

    models = [
        ('mbf',            'work_dirs/casia_mbf_arcface/model.pt'),
        ('shufflefacenet', 'work_dirs/casia_shuffle_arcface/model.pt'),
        ('vargfacenet',    'work_dirs/casia_vargface_arcface/model.pt'),
    ]

    for net, wpath in models:
        if not os.path.exists(wpath):
            print(f"\n[SKIP] {net}: {wpath} not found")
            continue
        print(f"\n{'='*70}")
        print(f"  {net.upper()} — Degraded Evaluation")
        print(f"{'='*70}")
        !python eval_degraded.py \
            --network {net} --weight {wpath} --rec {DATASET} \
            --degradations {DEGRADATIONS} --severities {SEVERITIES} --seed 42
else:
    print("SKIPPED (RUN_EVAL=False)")

# %% [markdown]
# ---
# ## Cell 13: Benchmark Model Efficiency

# %%
if RUN_BENCHMARK:
    os.chdir(WORK_DIR)
    import time, torch
    from backbones import get_model

    print("=" * 70)
    print("  MODEL EFFICIENCY BENCHMARK (T4 GPU)")
    print("=" * 70)

    x1 = torch.randn(1, 3, 112, 112).cuda()
    x16 = torch.randn(16, 3, 112, 112).cuda()

    header = (f"  {'Network':<20} {'Params(M)':>10} {'Size(MB)':>10} "
              f"{'GPU b=1(ms)':>14} {'GPU b=16(ms)':>14}")
    print(f"\n{header}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*14} {'-'*14}")

    for net in ['mbf', 'shufflefacenet', 'vargfacenet']:
        m = get_model(net, fp16=False, num_features=512).cuda().eval()
        params = sum(p.numel() for p in m.parameters()) / 1e6

        # Model size (FP32)
        tmp_path = '/tmp/_bench_model.pt'
        torch.save(m.state_dict(), tmp_path)
        size_mb = os.path.getsize(tmp_path) / 1024 / 1024
        os.remove(tmp_path)

        # GPU timing — batch=1
        with torch.no_grad():
            for _ in range(20): _ = m(x1)  # warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100): _ = m(x1)
            torch.cuda.synchronize()
            gpu_b1 = (time.perf_counter() - t0) / 100 * 1000

        # GPU timing — batch=16
        with torch.no_grad():
            for _ in range(10): _ = m(x16)  # warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30): _ = m(x16)
            torch.cuda.synchronize()
            gpu_b16 = (time.perf_counter() - t0) / 30 * 1000

        print(f"  {net:<20} {params:>10.2f} {size_mb:>10.1f} "
              f"{gpu_b1:>14.2f} {gpu_b16:>14.2f}")
        del m
    torch.cuda.empty_cache()
else:
    print("SKIPPED (RUN_BENCHMARK=False)")

# %% [markdown]
# ---
# ## Cell 14: Save Results to Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive')

import shutil

SAVE_DIR = '/content/drive/MyDrive/lightweight_fr_results'
os.makedirs(SAVE_DIR, exist_ok=True)

for d in ['casia_mbf_arcface', 'casia_shuffle_arcface', 'casia_vargface_arcface']:
    src = os.path.join(WORK_DIR, 'work_dirs', d)
    dst = os.path.join(SAVE_DIR, d)
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[OK] Saved: {dst}")
    else:
        print(f"[SKIP] {src} not found")

print(f"\nAll results saved to: {SAVE_DIR}")

# %% [markdown]
# ---
# ## Notes
#
# **Pretrained MobileFaceNet**: Chỉ dùng cho sanity check pipeline (Cell 9).
# So sánh công bằng PHẢI train tất cả backbone từ scratch cùng dataset + cùng setting.
#
# **Colab T4 không dùng để train full-scale**. CASIA-WebFace (490K images) là dataset
# vừa đủ cho research comparison. Kết quả accuracy sẽ thấp hơn so với MS1MV3.
#
# **Phase 2 (future)**: Sau khi chọn 1-2 backbone tốt nhất từ Phase 1,
# so sánh ArcFace vs AdaFace vs MagFace trên backbone đó.
