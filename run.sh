"""
For Recommendation Models
"""
# train teacher
python -u main.py --dataset=citeulike --S_backbone=bpr --train_teacher --suffix teacher
python -u main.py --dataset=citeulike --S_backbone=lightgcn --train_teacher --suffix teacher --postsave
python -u main.py --dataset=citeulike --S_backbone=hstu --train_teacher --suffix teacher --postsave

python -u main.py --dataset=gowalla --S_backbone=bpr --train_teacher --suffix teacher
python -u main.py --dataset=gowalla --S_backbone=lightgcn --train_teacher --suffix teacher --postsave
python -u main.py --dataset=gowalla --S_backbone=hstu --train_teacher --suffix teacher --postsave

python -u main.py --dataset=yelp --S_backbone=bpr --train_teacher --suffix teacher --suffix teacher
python -u main.py --dataset=yelp --S_backbone=lightgcn --train_teacher --suffix teacher --postsave
python -u main.py --dataset=yelp --S_backbone=hstu --train_teacher --suffix teacher --postsave

# from scratch
python -u main.py --dataset=citeulike --S_backbone=bpr --model=scratch --suffix student
python -u main.py --dataset=citeulike --S_backbone=lightgcn --model=scratch --suffix student
python -u main.py --dataset=citeulike --S_backbone=hstu --model=scratch --suffix student

python -u main.py --dataset=gowalla --S_backbone=bpr --model=scratch --suffix student
python -u main.py --dataset=gowalla --S_backbone=lightgcn --model=scratch --suffix student
python -u main.py --dataset=gowalla --S_backbone=hstu --model=scratch --suffix student

python -u main.py --dataset=yelp --S_backbone=bpr --model=scratch --suffix student
python -u main.py --dataset=yelp --S_backbone=lightgcn --model=scratch --suffix student
python -u main.py --dataset=yelp --S_backbone=hstu --model=scratch --suffix student

# KD
# For HetComp, you need pre-save teacher checkpoints through:
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --train_teacher --no_log --ckpt_interval=50
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=hetcomp
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=de
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=rrd
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=dcd
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=rcekd

python -u main.py --dataset=citeulike --S_backbone=lightgcn --T_backbone=lightgcn --model=de
python -u main.py --dataset=citeulike --S_backbone=lightgcn --T_backbone=lightgcn --model=rrd


python -u main.py --dataset=gowalla --S_backbone=bpr --T_backbone=bpr --model=de
python -u main.py --dataset=gowalla --S_backbone=bpr --T_backbone=bpr --model=rrd


python -u main.py --dataset=yelp --S_backbone=bpr --T_backbone=bpr --model=de
python -u main.py --dataset=yelp --S_backbone=bpr --T_backbone=bpr --model=rrd


# ===== SRCE-KD (our improvement) pre-research =====
# stage 1: diagnostics (no training; needs teacher + rcekd student checkpoints)
# optional: rerun rcekd with intermediate checkpoints to get overlap/blind-fraction dynamics
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=rcekd --ckpt_interval=100 --postsave --suffix dyn
python -u diagnose.py --dataset=citeulike --T_backbone=bpr --S_backbone=bpr --model=rcekd
python -u diagnose.py --dataset=citeulike --T_backbone=bpr --S_backbone=bpr --model=rcekd --suffix dyn

# stage 2: effectiveness check of SRCE-KD (1 seed first, then --run_all for 5 seeds)
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=srcekd
python -u main.py --dataset=gowalla --S_backbone=lightgcn --T_backbone=lightgcn --model=srcekd

# stage 3: split mode (RCE-KD + uniform tail coverage, the main candidate)
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=srcekd --cfg srce_mode=split srce_Lu=20 early_stop_K=20 --suffix splitLu20_homo_seed0
python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=lightgcn --model=srcekd --cfg srce_mode=split srce_Lu=20 early_stop_K=20 --suffix splitLu20_het_seed0
