<!-- ============================================================
     MPI-INF Cluster Setup & Training Guide
     Submit host: ssh slurm-submit.mpi-inf.mpg.de
     ============================================================ -->

<!-- STEP 1: Login to submit host -->
ssh slurm-submit.mpi-inf.mpg.de

<!-- STEP 2: Install Miniforge3 (once) -->
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"

<!-- STEP 3: Enable conda + configure conda-forge (once) -->
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda config --set channel_priority strict
conda config --add channels conda-forge
conda config --remove channels defaults || true

<!-- Add to ~/.bashrc so conda is always available -->
echo 'source "$HOME/miniforge3/etc/profile.d/conda.sh"' >> ~/.bashrc

<!-- STEP 4: Create project environment (once) -->
conda create -n smoothing python=3.11 pip -y
conda activate smoothing

<!-- STEP 5: Install project dependencies -->
cd /BS/dniazi_thesis/work/RandomizedSmothingmanifold
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

<!-- ============================================================
     Submit Training Jobs (from slurm-submit.mpi-inf.mpg.de)
     ============================================================ -->

<!-- CelebA -->
sbatch src/experiments/training/resnet_smile/submit_slurm.sh src/configs/training/smile_resnet_celeba.yaml

<!-- CelebA-HQ -->
sbatch src/experiments/training/resnet_smile/submit_slurm.sh src/configs/training/smile_resnet_celebahq.yaml

<!-- Monitor -->
squeue -u $USER
cat output/slurm/smile-<jobid>.out

<!-- Cancel -->
scancel <jobid>

<!-- ============================================================
     Interactive GPU session (testing only, 1h limit)
     ============================================================ -->

gpusession start
<!-- or -->
srun -p gpu20 --gres gpu:1 -c 4 --mem-per-cpu=4G --pty /bin/bash
<!-- inside the session -->
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate smoothing
cd /BS/dniazi_thesis/work/RandomizedSmothingmanifold
python -m src.experiments.training.resnet_smile.main --config src/configs/training/smile_resnet_celeba.yaml

<!-- ============================================================
 SLURM commands
     ============================================================ -->

sinfo -rO partition,nodelist,gres:30,gresused:30   <!-- show GPU availability -->
squeue -j 48293049                                     <!--STATUS -->
scontrol show job <jobid>                           <!-- full job info -->
srun --jobid <jobid> --pty bash                    <!-- attach second shell to running job -->
tail -f output/slurm/smoke-48293049.out                 <!--Out logs -->

<!-- ============================================================
     Output locations
     ============================================================ -->

<!-- Checkpoints (latest 2 kept):
     output/pretrained_model/CelebA/resnet18_checkpoint_epoch_<N>.pt
     output/pretrained_model/CelebA/resnet18_latest.pt              -->

<!-- Logs:
     output/training_history.csv
     output/training_summary.json
     output/slurm/smile-<jobid>.out                                  -->

<!-- RUN Basic .py file -->
ssh slurm-submit
cd /BS/dniazi_thesis/work/RandomizedSmothingmanifold
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate smoothing
python test.py