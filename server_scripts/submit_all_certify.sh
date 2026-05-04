#!/usr/bin/env bash
# Submit all CelebA certification experiments
# Usage: bash server_scripts/submit_all_certify.sh

set -euo pipefail

echo "Submitting CelebA certification experiments..."

# CelebA Pixel
echo "Submitting CelebA Pixel certification..."
JOB1=$(sbatch server_scripts/submit_certify_celeba_pixel.sh | awk '{print $4}')
echo "  Job ID: $JOB1"

# CelebA-HQ Pixel
echo "Submitting CelebA-HQ Pixel certification..."
JOB2=$(sbatch server_scripts/submit_certify_celebahq_pixel.sh | awk '{print $4}')
echo "  Job ID: $JOB2"

# CelebA Latent (depends on VAE training)
echo "Submitting CelebA Latent certification..."
JOB3=$(sbatch server_scripts/submit_certify_celeba_latent.sh | awk '{print $4}')
echo "  Job ID: $JOB3"

# CelebA-HQ Latent (depends on VAE training)
echo "Submitting CelebA-HQ Latent certification..."
JOB4=$(sbatch server_scripts/submit_certify_celebahq_latent.sh | awk '{print $4}')
echo "  Job ID: $JOB4"

echo ""
echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Jobs:"
echo "  $JOB1 - CelebA Pixel"
echo "  $JOB2 - CelebA-HQ Pixel"
echo "  $JOB3 - CelebA Latent"
echo "  $JOB4 - CelebA-HQ Latent"
