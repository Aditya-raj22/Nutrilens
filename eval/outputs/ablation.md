# Ablation Study

| arch     | frozen   | augmented   |   best_val_acc |   epochs_trained | tag                   |
|:---------|:---------|:------------|---------------:|-----------------:|:----------------------|
| resnet50 | False    | True        |         0.8286 |               15 | resnet50              |
| resnet50 | True     | False       |         0.5737 |                8 | resnet50_frozen_noaug |
| resnet50 | True     | True        |         0.572  |                8 | resnet50_frozen       |
| vit_b_16 | False    | True        |         0.819  |               20 | vit_b_16              |
| vit_b_16 | True     | True        |         0.6995 |                8 | vit_b_16_frozen       |
