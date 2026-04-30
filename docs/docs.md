Project setup (Linux/macOS):

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Optional notebook kernel:

python -m ipykernel install --user --name randomizedsmoothingmanifold --display-name "Python (.venv)"

Run NER manifold smoothing baseline (CoNLL-2003 + DistilBERT):

python -m src.eval.run_ner_experiment --config src/configs/experiments/ner_conll2003_distilbert.yaml

Main outputs are written to the configured output directory:

- resolved_config.yaml
- history.json
- metrics.json
- model.pt
