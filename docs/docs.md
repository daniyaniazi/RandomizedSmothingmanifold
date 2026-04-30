Project setup (Linux/macOS):

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

Optional notebook kernel:

python3 -m ipykernel install --user --name randomizedsmoothingmanifold --display-name "Python (.venv)"

Run NER manifold smoothing baseline (CoNLL-2003 + DistilBERT):

python3 -m src.eval.run_ner_experiment --config src/configs/experiments/ner_conll2003_distilbert.yaml

Run ResNet smile training (CelebA/CelebA-HQ):

python3 -m src.experiments.training.resnet_smile.main --config src/configs/training/smile_resnet_celeba.yaml

python3 -m src.experiments.training.resnet_smile.main --config src/configs/training/smile_resnet_celebahq.yaml

SLURM submit example:

sbatch src/experiments/training/resnet_smile/submit_slurm.sh src/configs/training/smile_resnet_celeba.yaml

Main outputs are written to the configured output directory:

- resolved_config.yaml
- history.json
- metrics.json
- model.pt

Smile model checkpoints are written to:

- output/pretrained_model/<dataset_name>/<model_name>_checkpoint_epoch_<N>.pt

Only the latest 2 epoch checkpoints are retained automatically.
