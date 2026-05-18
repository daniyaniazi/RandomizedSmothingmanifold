#!/usr/bin/env bash
# ============================================================================
# MASTER CERTIFICATION EXPERIMENT RUNNER
# ============================================================================
# Submit all certification experiments to SLURM cluster
#
# Usage:
#   ./submit_all_certify.sh [OPTIONS]
#
# Options:
#   --celeba     Submit all CelebA experiments only
#   --celebahq   Submit all CelebA-HQ experiments only
#   --ner        Submit all NER experiments only
#   --isotropic  Submit isotropic (Gaussian) baseline experiments only
#   --manifold   Submit manifold-aware experiments only
#   --dry-run    Print commands without submitting
#   --help       Show this help message
#
# Experiment Matrix:
# ┌─────────────┬────────────┬────────────┬──────────────────────────────────────┐
# │ Domain      │ Smoothing  │ Variant    │ Script                               │
# ├─────────────┼────────────┼────────────┼──────────────────────────────────────┤
# │ CelebA      │ Isotropic  │ pixel      │ submit_certify_celeba_isotropic.sh   │
# │ CelebA      │ Manifold   │ pixel      │ submit_certify_celeba_pixel.sh       │
# │ CelebA      │ Manifold   │ latent     │ submit_certify_celeba_latent.sh      │
# ├─────────────┼────────────┼────────────┼──────────────────────────────────────┤
# │ CelebA-HQ   │ Isotropic  │ pixel      │ submit_certify_celebahq_isotropic.sh │
# │ CelebA-HQ   │ Manifold   │ pixel      │ submit_certify_celebahq_pixel.sh     │
# │ CelebA-HQ   │ Manifold   │ latent     │ submit_certify_celebahq_latent.sh    │
# ├─────────────┼────────────┼────────────┼──────────────────────────────────────┤
# │ NER         │ Isotropic  │ hidden     │ submit_ner_bert_isotropic_certify.sh │
# │ NER         │ Manifold   │ hidden     │ submit_ner_bert_certify.sh           │
# │ NER+Masking │ Isotropic  │ hidden     │ submit_ner_bert_isotropic_masking... │
# │ NER+Masking │ Manifold   │ hidden     │ submit_ner_bert_masking_certify.sh   │
# └─────────────┴────────────┴────────────┴──────────────────────────────────────┘
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default: submit all
SUBMIT_CELEBA=true
SUBMIT_CELEBAHQ=true
SUBMIT_NER=true
SUBMIT_ISOTROPIC=true
SUBMIT_MANIFOLD=true
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --celeba)
            SUBMIT_CELEBAHQ=false
            SUBMIT_NER=false
            shift
            ;;
        --celebahq)
            SUBMIT_CELEBA=false
            SUBMIT_NER=false
            shift
            ;;
        --ner)
            SUBMIT_CELEBA=false
            SUBMIT_CELEBAHQ=false
            shift
            ;;
        --isotropic)
            SUBMIT_MANIFOLD=false
            shift
            ;;
        --manifold)
            SUBMIT_ISOTROPIC=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            head -50 "$0" | tail -n +2
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

submit_job() {
    local script=$1
    local desc=$2
    
    if [[ ! -f "$script" ]]; then
        echo "⚠️  Script not found: $script"
        return
    fi
    
    if $DRY_RUN; then
        echo "🔹 [DRY-RUN] Would submit: $desc"
        echo "   sbatch $script"
    else
        echo "🚀 Submitting: $desc"
        sbatch "$script"
    fi
}

echo "=============================================="
echo "CERTIFICATION EXPERIMENT SUBMISSION"
echo "=============================================="
echo ""

# ============================================================================
# CelebA Experiments
# ============================================================================
if $SUBMIT_CELEBA; then
    echo "📁 CelebA Experiments"
    echo "---------------------"
    
    if $SUBMIT_ISOTROPIC; then
        submit_job "submit_certify_celeba_isotropic.sh" \
            "CelebA Isotropic (Gaussian) - Pixel Space"
        submit_job "submit_certify_celeba_isotropic_latent.sh" \
            "CelebA Isotropic (Gaussian) - Latent Space (VAE)"
    fi
    
    if $SUBMIT_MANIFOLD; then
        submit_job "submit_certify_celeba_pixel.sh" \
            "CelebA Manifold - Pixel Space"
        submit_job "submit_certify_celeba_latent.sh" \
            "CelebA Manifold - Latent Space (VAE)"
    fi
    echo ""
fi

# ============================================================================
# CelebA-HQ Experiments
# ============================================================================
if $SUBMIT_CELEBAHQ; then
    echo "📁 CelebA-HQ Experiments"
    echo "------------------------"
    
    if $SUBMIT_ISOTROPIC; then
        submit_job "submit_certify_celebahq_isotropic.sh" \
            "CelebA-HQ Isotropic (Gaussian) - Pixel Space"
        submit_job "submit_certify_celebahq_isotropic_latent.sh" \
            "CelebA-HQ Isotropic (Gaussian) - Latent Space (VAE)"
    fi
    
    if $SUBMIT_MANIFOLD; then
        submit_job "submit_certify_celebahq_pixel.sh" \
            "CelebA-HQ Manifold - Pixel Space"
        submit_job "submit_certify_celebahq_latent.sh" \
            "CelebA-HQ Manifold - Latent Space (VAE)"
    fi
    echo ""
fi

# ============================================================================
# NER Experiments
# ============================================================================
if $SUBMIT_NER; then
    echo "📁 NER Experiments (BERT + CoNLL-2003)"
    echo "--------------------------------------"
    
    if $SUBMIT_ISOTROPIC; then
        submit_job "submit_ner_bert_isotropic_certify.sh" \
            "NER Isotropic (Gaussian) - Hidden States"
        submit_job "submit_ner_bert_isotropic_masking_certify.sh" \
            "NER Isotropic + Context Masking"
    fi
    
    if $SUBMIT_MANIFOLD; then
        submit_job "submit_ner_bert_certify.sh" \
            "NER Manifold - Hidden States"
        submit_job "submit_ner_bert_masking_certify.sh" \
            "NER Manifold + Context Masking"
    fi
    echo ""
fi

echo "=============================================="
if $DRY_RUN; then
    echo "DRY RUN COMPLETE - No jobs submitted"
else
    echo "ALL JOBS SUBMITTED"
    echo "Check status: squeue -u \$USER"
fi
echo "=============================================="
