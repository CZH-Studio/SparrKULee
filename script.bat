python -m match_mismatch.src.experiments.sota data.dataloader.preserve_ratio=0.1 name=p10nc5_seed-2 seed=2
python -m match_mismatch.src.experiments.sota data.dataloader.preserve_ratio=0.1 name=p10nc5_seed-3 seed=3
python -m match_mismatch.src.experiments.sota data.dataloader.preserve_ratio=0.1 name=p10nc5_seed-4 seed=4
python -m match_mismatch.src.experiments.sota data.dataloader.preserve_ratio=0.1 name=p10nc5_seed-5 seed=5

python -m match_mismatch.src.experiments.clip_cls data.dataloader.preserve_ratio=0.1 name=p10nc5_seed-1 seed=1 
python -m match_mismatch.src.experiments.clip_cls data.dataloader.preserve_ratio=0.1 name=p10nc5_seed-2 seed=2 
python -m match_mismatch.src.experiments.clip_cls data.dataloader.preserve_ratio=0.1 name=p10nc5_seed-3 seed=3 
python -m match_mismatch.src.experiments.clip_cls data.dataloader.preserve_ratio=0.1 name=p10nc5_seed-4 seed=4 
python -m match_mismatch.src.experiments.clip_cls data.dataloader.preserve_ratio=0.1 name=p10nc5_seed-5 seed=5 

python -m match_mismatch.src.experiments.clip_cls data.dataloader.preserve_ratio=0.1 name=p10nc5_seed-1 seed=1 model.clip.freeze_grad_when_tuning=false
