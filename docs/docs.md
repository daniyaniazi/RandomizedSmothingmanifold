py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

python -m pip install jupyter ipykernel kaggle torch torchvision matplotlib numpy scikit-learn scipy tqdm annoy
python -m ipykernel install --user --name randomizedsmoothingmanifold --display-name "Python (.venv)"
