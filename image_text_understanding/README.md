# Finetuning VL-BART on image-text understaing tasks using MTL-LoRA

We evaluate MTL-LoRA in a unified multi-task
setup on image-text following the settings of VL-Adapter and DoRA. We use four diverse V&L datasets: VQAv2, GQA, NLVR2, and MSCOCO image captioning.

## Setup
```
# Create python environment
conda create -n vlt5 python=3.8
source activate vlt5

# Install python dependencies
pip install -r requirements.txt

# Download T5/BART backbone checkpoint
python download_backbones.py

# For MSCOCO captioning evaluation (optional; for captioning only)
python -c "import language_evaluation; language_evaluation.download('coco')"
```

## Data
```bash
# Store images, features, and annotations
./datasets
    COCO/
        images/
        clip_featuers/
    VG/
        images/
        clip_features/
    GQA/
        images/
        clip_features/
    nlvr/
        images/
        clip_features/
    vqa/
    lxmert/

# Train VL-T5 with adapters
./VL-T5/
    src/
        multitask.py    <= multitask learning on 7 downstream tasks
        trainer_base.py <= DoRA implementation
```

### Image-text dataset
Please go to [link](https://drive.google.com/file/d/1O_RU1iFh_sbItZCTkOHUrbVIQQ_89Djj/view?usp=sharing) to download the processed CLIP features. We suggest to use [gdrive](https://github.com/prasmussen/gdrive) to download it. Unzip the downloaded file and arrange the folders according to the format demonstrated above.

If you would like to use dgrive to download the data, please try the following command

```
gdrive download 1O_RU1iFh_sbItZCTkOHUrbVIQQ_89Djj
```

## Finetuning and Evaluation
### Finetuning VL-BART on Image-text datasets with DoRA (Evaluation included)
```
bash ./VL-T5/scripts/image/mlora.sh 1
```

## Acknowledgement
We greatly appreciate the contributions of [VL-Adapter](https://github.com/ylsung/VL_adapter) and [DoRA](https://github.com/NVlabs/DoRA) which has significantly benefited our work.