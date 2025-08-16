@echo off
for /l %%i in (7,1,20) do (
    echo Running: python -m match_mismatch.src.experiments.clip_cls^
 data.subset_ratio=0.1^
 model.clip.pretrained_model_path=^
E:/Code/SparrKULee/SparrKULee/match_mismatch/output/clip_pretrained/bsz-32_easy/models/model.ckpt^
 seed=%%i^
 model.clip.freeze_grad_when_tuning=true
    python -m match_mismatch.src.experiments.clip_cls^
 data.subset_ratio=0.1^
 model.clip.pretrained_model_path=^
E:/Code/SparrKULee/SparrKULee/match_mismatch/output/clip_pretrained/bsz-32_easy/models/model.ckpt^
 seed=%%i^
 model.clip.freeze_grad_when_tuning=true
)
pause