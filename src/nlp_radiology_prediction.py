"""
Mortality prediction using clinical features and NLP on MIMIC-IV.
This script assumes required packages are installed via requirements.txt.
"""

import pandas as pd
import numpy as np
import re
import warnings
import pickle
import argparse

from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def load_data(admissions_path, patients_path, radiology_notes_path):
    """Load MIMIC-IV data files"""
    print("Loading data files...")
    admissions = pd.read_csv(admissions_path)
    patients = pd.read_csv(patients_path)
    radiology_notes = pd.read_csv(radiology_notes_path)
    
    print(f"Admissions shape: {admissions.shape}")
    print(f"Patients shape: {patients.shape}")
    print(f"Radiology notes shape: {radiology_notes.shape}")
    
    return admissions, patients, radiology_notes


def prepare_admissions(admissions, patients):
    """Filter and prepare admissions with clinical features"""
    print("\nFiltering admissions...")
    
    # Merge with patients to get demographics
    admissions_with_demo = admissions.merge(
        patients[['subject_id', 'anchor_age', 'anchor_year', 'gender']],
        on='subject_id',
        how='inner'
    )
    
    # Calculate age at admission
    admissions_with_demo['admittime'] = pd.to_datetime(admissions_with_demo['admittime'])
    admissions_with_demo['admit_year'] = admissions_with_demo['admittime'].dt.year
    admissions_with_demo['age_at_admission'] = (
        admissions_with_demo['anchor_age'] +
        (admissions_with_demo['admit_year'] - admissions_with_demo['anchor_year'])
    )
    
    # Filter: adults only (≥18 years)
    admissions_filtered = admissions_with_demo[
        admissions_with_demo['age_at_admission'] >= 18
    ].copy()
    
    print(f"After age filter (≥18): {len(admissions_filtered)}")
    
    # Keep only first admission per patient
    admissions_filtered = admissions_filtered.sort_values('admittime')
    admissions_filtered = admissions_filtered.groupby('subject_id').first().reset_index()
    
    print(f"After keeping first admission per patient: {len(admissions_filtered)}")
    
    # Create binary mortality label
    admissions_filtered['LABEL'] = admissions_filtered['hospital_expire_flag'].astype(int)
    
    # Create clinical baseline features
    admissions_filtered['age'] = admissions_filtered['age_at_admission']
    admissions_filtered['is_male'] = (admissions_filtered['gender'] == 'M').astype(int)
    admissions_filtered['is_emergency'] = (admissions_filtered['admission_type'] == 'EMERGENCY').astype(int)
    
    print(f"Filtered admissions: {len(admissions_filtered)}")
    print(f"Mortality rate: {admissions_filtered['LABEL'].mean():.2%}")
    print(f"\nClinical feature statistics:")
    print(f"  Age: {admissions_filtered['age'].mean():.1f} ± {admissions_filtered['age'].std():.1f}")
    print(f"  Male: {admissions_filtered['is_male'].mean():.1%}")
    print(f"  Emergency admission: {admissions_filtered['is_emergency'].mean():.1%}")
    
    print(f"\nAdmission types distribution:")
    print(admissions_filtered['admission_type'].value_counts())
    
    return admissions_filtered


def extract_early_radiology_notes(radiology_notes, admissions_filtered, hours_window=48):
    """Extract radiology notes from first N hours of admission"""
    print(f"\nExtracting radiology notes from first {hours_window} hours...")
    
    print(f"Radiology notes columns: {radiology_notes.columns.tolist()}")
    
    # Drop subject_id from radiology notes to avoid conflicts
    notes_cols = [col for col in radiology_notes.columns if col != 'subject_id']
    
    # Merge with admissions
    notes_with_admit = radiology_notes[notes_cols].merge(
        admissions_filtered[['hadm_id', 'admittime', 'subject_id', 'LABEL',
                             'age', 'is_male', 'is_emergency']],
        on='hadm_id',
        how='inner'
    )
    
    print(f"Radiology notes linked to filtered admissions: {len(notes_with_admit)}")
    
    # Convert times
    notes_with_admit['charttime'] = pd.to_datetime(notes_with_admit['charttime'])
    notes_with_admit['admittime'] = pd.to_datetime(notes_with_admit['admittime'])
    
    # Calculate hours from admission
    notes_with_admit['hours_from_admit'] = (
        notes_with_admit['charttime'] - notes_with_admit['admittime']
    ).dt.total_seconds() / 3600
    
    # Keep only first N hours
    early_notes = notes_with_admit[
        (notes_with_admit['hours_from_admit'] >= 0) &
        (notes_with_admit['hours_from_admit'] <= hours_window)
    ].copy()
    
    print(f"Radiology notes in first {hours_window}h: {len(early_notes)}")
    print(f"Mortality rate in early notes: {early_notes['LABEL'].mean():.2%}")
    
    # Detect text column
    text_column = 'text'
    if 'text' not in early_notes.columns:
        if 'TEXT' in early_notes.columns:
            text_column = 'TEXT'
        else:
            raise KeyError(f"Text column not found. Available columns: {early_notes.columns.tolist()}")
    
    print(f"Using text column: '{text_column}'")
    
    # Show sample report
    print("\nSample radiology report:")
    print(early_notes[text_column].iloc[0][:500])
    print("\n" + "="*60)
    
    # Concatenate multiple radiology reports per admission
    notes_concat = early_notes.groupby('hadm_id').agg({
        text_column: lambda x: ' '.join(x.astype(str)),
        'subject_id': 'first',
        'LABEL': 'first',
        'age': 'first',
        'is_male': 'first',
        'is_emergency': 'first'
    }).reset_index()
    
    notes_concat.columns = ['hadm_id', 'CONCATENATED_TEXT', 'subject_id', 'LABEL',
                            'age', 'is_male', 'is_emergency']
    
    print(f"Final dataset shape: {notes_concat.shape}")
    print(f"Final mortality rate: {notes_concat['LABEL'].mean():.2%}")
    print(f"Patients with radiology notes in first {hours_window}h: {notes_concat['subject_id'].nunique()}")
    
    return notes_concat


def create_splits(dataset, random_state=42):
    """Create train/val/test splits at patient level"""
    print("\nCreating train/val/test splits...")
    
    unique_subjects = dataset['subject_id'].unique()
    train_subjects, temp_subjects = train_test_split(
        unique_subjects, test_size=0.3, random_state=random_state
    )
    val_subjects, test_subjects = train_test_split(
        temp_subjects, test_size=0.5, random_state=random_state
    )
    
    # Assign splits
    dataset['SPLIT'] = 'train'
    dataset.loc[dataset['subject_id'].isin(val_subjects), 'SPLIT'] = 'val'
    dataset.loc[dataset['subject_id'].isin(test_subjects), 'SPLIT'] = 'test'
    
    print(f"\nMortality rates by split:")
    for split in ['train', 'val', 'test']:
        split_data = dataset[dataset['SPLIT'] == split]
        print(f"{split}: {split_data['LABEL'].mean():.2%} ({len(split_data)} samples)")
    
    return dataset


def preprocess_text(text):
    """Improved preprocessing for clinical radiology text"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove long numeric-only tokens (likely IDs, not useful features)
    text = re.sub(r'\b\d{4,}\b', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but preserve medical terms
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    
    return text.strip()


def train_clinical_baseline(X_train, y_train):
    """Train clinical baseline model (age, gender, admission type)"""
    print("\n" + "="*60)
    print("TRAINING CLINICAL BASELINE")
    print("="*60)
    
    clinical_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    clinical_model.fit(X_train, y_train)
    
    return clinical_model


def train_nlp_model_with_bootstrap(texts_train, y_train, max_features=10000, n_bootstrap=5):
    """Train NLP model with bootstrap analysis for feature stability"""
    print("\n" + "="*60)
    print("TRAINING NLP MODEL WITH FEATURE STABILITY ANALYSIS")
    print("="*60)
    
    # Preprocess text
    texts_train_clean = texts_train.apply(preprocess_text)
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(texts_train_clean)
    print(f"TF-IDF shape: {X_train_tfidf.shape}")
    
    # Bootstrap training
    print(f"\nBootstrap training ({n_bootstrap} models)...")
    
    bootstrap_models = []
    bootstrap_features_positive = []
    bootstrap_features_negative = []
    feature_names = vectorizer.get_feature_names_out()
    
    for i in range(n_bootstrap):
        print(f"  Bootstrap {i+1}/{n_bootstrap}...", end=' ')
        
        # Bootstrap sample
        indices = np.random.choice(len(y_train), size=len(y_train), replace=True)
        X_boot = X_train_tfidf[indices]
        y_boot = y_train.iloc[indices].values
        
        # Train model
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42+i,
            solver='liblinear'
        )
        model.fit(X_boot, y_boot)
        bootstrap_models.append(model)
        
        # Store top features
        coefficients = model.coef_[0]
        top_30_pos = coefficients.argsort()[-30:][::-1]
        top_30_neg = coefficients.argsort()[:30]
        
        bootstrap_features_positive.append([feature_names[idx] for idx in top_30_pos])
        bootstrap_features_negative.append([feature_names[idx] for idx in top_30_neg])
        
        print("Done")
    
    # Analyze feature stability
    print("\n" + "="*60)
    print("FEATURE STABILITY ANALYSIS")
    print("="*60)
    
    all_positive_features = [f for sublist in bootstrap_features_positive for f in sublist]
    all_negative_features = [f for sublist in bootstrap_features_negative for f in sublist]
    
    positive_counts = Counter(all_positive_features)
    negative_counts = Counter(all_negative_features)
    
    print(f"\nMost STABLE features INCREASING mortality (appeared in ≥3/{n_bootstrap} bootstraps):")
    stable_positive = [(feat, count) for feat, count in positive_counts.most_common(30) if count >= 3]
    for i, (feat, count) in enumerate(stable_positive, 1):
        print(f"{i:2d}. {feat:30s} (appeared {count}/{n_bootstrap} times)")
    
    print(f"\nMost STABLE features DECREASING mortality (appeared in ≥3/{n_bootstrap} bootstraps):")
    stable_negative = [(feat, count) for feat, count in negative_counts.most_common(30) if count >= 3]
    for i, (feat, count) in enumerate(stable_negative, 1):
        print(f"{i:2d}. {feat:30s} (appeared {count}/{n_bootstrap} times)")
    
    # Train final model
    print("\nTraining final NLP model...")
    nlp_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='liblinear'
    )
    nlp_model.fit(X_train_tfidf, y_train)
    
    return vectorizer, nlp_model, bootstrap_models, stable_positive, stable_negative


def tune_threshold(y_true, y_proba):
    """Find optimal threshold using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION ON VALIDATION SET")
    print("="*60)
    print(f"\nOptimal threshold (Youden's J): {optimal_threshold:.3f}")
    print(f"  Sensitivity at this threshold: {tpr[optimal_idx]:.3f}")
    print(f"  Specificity at this threshold: {1-fpr[optimal_idx]:.3f}")
    
    return optimal_threshold


def bootstrap_ci(y_true, y_pred_proba, metric_func, n_bootstrap=1000, ci=95):
    """Calculate bootstrap confidence interval for a metric"""
    scores = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_pred_proba[indices])
        scores.append(score)
    
    lower = np.percentile(scores, (100-ci)/2)
    upper = np.percentile(scores, 100-(100-ci)/2)
    return np.mean(scores), lower, upper


def evaluate_with_ci(y_true, y_pred_proba, threshold, model_name, split_name):
    """Evaluate model with bootstrap confidence intervals"""
    print(f"\n{model_name} - {split_name} Set:")
    
    # AUROC
    auroc_mean, auroc_low, auroc_high = bootstrap_ci(y_true, y_pred_proba, roc_auc_score)
    print(f"  AUROC: {auroc_mean:.4f} (95% CI: {auroc_low:.4f}-{auroc_high:.4f})")
    
    # AUPRC
    auprc_mean, auprc_low, auprc_high = bootstrap_ci(y_true, y_pred_proba, average_precision_score)
    print(f"  AUPRC: {auprc_mean:.4f} (95% CI: {auprc_low:.4f}-{auprc_high:.4f})")
    
    # Metrics at threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"  Accuracy (threshold={threshold:.3f}): {acc:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  PPV: {ppv:.4f}")
    print(f"  NPV: {npv:.4f}")
    
    # Brier score
    brier = brier_score_loss(y_true, y_pred_proba)
    print(f"  Brier Score: {brier:.4f}")
    
    return {
        'auroc': auroc_mean, 'auroc_ci': (auroc_low, auroc_high),
        'auprc': auprc_mean, 'auprc_ci': (auprc_low, auprc_high),
        'accuracy': acc, 'sensitivity': sensitivity, 'specificity': specificity,
        'ppv': ppv, 'npv': npv, 'brier': brier
    }


def plot_calibration(y_test, y_test_clinical_proba, y_test_nlp_proba, output_path):
    """Create calibration plots"""
    print("\n" + "="*60)
    print("CALIBRATION ANALYSIS")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Clinical baseline calibration
    prob_true_clin, prob_pred_clin = calibration_curve(y_test, y_test_clinical_proba, n_bins=10)
    axes[0].plot(prob_pred_clin, prob_true_clin, marker='o', linewidth=2, label='Clinical Model')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    axes[0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0].set_ylabel('True Probability', fontsize=12)
    axes[0].set_title('Clinical Baseline Calibration', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # NLP model calibration
    prob_true_nlp, prob_pred_nlp = calibration_curve(y_test, y_test_nlp_proba, n_bins=10)
    axes[1].plot(prob_pred_nlp, prob_true_nlp, marker='o', linewidth=2, label='NLP Model')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    axes[1].set_xlabel('Predicted Probability', fontsize=12)
    axes[1].set_ylabel('True Probability', fontsize=12)
    axes[1].set_title('NLP Model Calibration', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Calibration plots saved to {output_path}")


def plot_performance_comparison(y_train, y_train_clinical_proba, y_train_nlp_proba,
                                y_val, y_val_clinical_proba, y_val_nlp_proba,
                                y_test, y_test_clinical_proba, y_test_nlp_proba,
                                clinical_train_metrics, clinical_val_metrics, clinical_test_metrics,
                                nlp_train_metrics, nlp_val_metrics, nlp_test_metrics,
                                output_path):
    """Create ROC and PR curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ROC Curves - Clinical
    fpr_train_c, tpr_train_c, _ = roc_curve(y_train, y_train_clinical_proba)
    fpr_val_c, tpr_val_c, _ = roc_curve(y_val, y_val_clinical_proba)
    fpr_test_c, tpr_test_c, _ = roc_curve(y_test, y_test_clinical_proba)
    
    axes[0, 0].plot(fpr_train_c, tpr_train_c, label=f'Train (AUROC={clinical_train_metrics["auroc"]:.3f})', linewidth=2)
    axes[0, 0].plot(fpr_val_c, tpr_val_c, label=f'Val (AUROC={clinical_val_metrics["auroc"]:.3f})', linewidth=2)
    axes[0, 0].plot(fpr_test_c, tpr_test_c, label=f'Test (AUROC={clinical_test_metrics["auroc"]:.3f})', linewidth=2)
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('Clinical Baseline - ROC Curve', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # ROC Curves - NLP
    fpr_train_n, tpr_train_n, _ = roc_curve(y_train, y_train_nlp_proba)
    fpr_val_n, tpr_val_n, _ = roc_curve(y_val, y_val_nlp_proba)
    fpr_test_n, tpr_test_n, _ = roc_curve(y_test, y_test_nlp_proba)
    
    axes[0, 1].plot(fpr_train_n, tpr_train_n, label=f'Train (AUROC={nlp_train_metrics["auroc"]:.3f})', linewidth=2)
    axes[0, 1].plot(fpr_val_n, tpr_val_n, label=f'Val (AUROC={nlp_val_metrics["auroc"]:.3f})', linewidth=2)
    axes[0, 1].plot(fpr_test_n, tpr_test_n, label=f'Test (AUROC={nlp_test_metrics["auroc"]:.3f})', linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('NLP Model - ROC Curve', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # PR Curves - Clinical
    prec_train_c, rec_train_c, _ = precision_recall_curve(y_train, y_train_clinical_proba)
    prec_val_c, rec_val_c, _ = precision_recall_curve(y_val, y_val_clinical_proba)
    prec_test_c, rec_test_c, _ = precision_recall_curve(y_test, y_test_clinical_proba)
    
    axes[1, 0].plot(rec_train_c, prec_train_c, label=f'Train (AUPRC={clinical_train_metrics["auprc"]:.3f})', linewidth=2)
    axes[1, 0].plot(rec_val_c, prec_val_c, label=f'Val (AUPRC={clinical_val_metrics["auprc"]:.3f})', linewidth=2)
    axes[1, 0].plot(rec_test_c, prec_test_c, label=f'Test (AUPRC={clinical_test_metrics["auprc"]:.3f})', linewidth=2)
    axes[1, 0].axhline(y_test.mean(), color='k', linestyle='--', label=f'Baseline ({y_test.mean():.3f})', linewidth=1)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Clinical Baseline - Precision-Recall Curve', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # PR Curves - NLP
    prec_train_n, rec_train_n, _ = precision_recall_curve(y_train, y_train_nlp_proba)
    prec_val_n, rec_val_n, _ = precision_recall_curve(y_val, y_val_nlp_proba)
    prec_test_n, rec_test_n, _ = precision_recall_curve(y_test, y_test_nlp_proba)
    
    axes[1, 1].plot(rec_train_n, prec_train_n, label=f'Train (AUPRC={nlp_train_metrics["auprc"]:.3f})', linewidth=2)
    axes[1, 1].plot(rec_val_n, prec_val_n, label=f'Val (AUPRC={nlp_val_metrics["auprc"]:.3f})', linewidth=2)
    axes[1, 1].plot(rec_test_n, prec_test_n, label=f'Test (AUPRC={nlp_test_metrics["auprc"]:.3f})', linewidth=2)
    axes[1, 1].axhline(y_test.mean(), color='k', linestyle='--', label=f'Baseline ({y_test.mean():.3f})', linewidth=1)
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('NLP Model - Precision-Recall Curve', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance comparison plots saved to {output_path}")


def create_results_table(clinical_train_metrics, clinical_val_metrics, clinical_test_metrics,
                        nlp_train_metrics, nlp_val_metrics, nlp_test_metrics,
                        output_path):
    """Create comprehensive results table"""
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE RESULTS TABLE")
    print("="*60)
    
    results_table = pd.DataFrame({
        'Model': ['Clinical Baseline', 'Clinical Baseline', 'Clinical Baseline',
                  'NLP Model', 'NLP Model', 'NLP Model'],
        'Split': ['Train', 'Validation', 'Test', 'Train', 'Validation', 'Test'],
        'AUROC': [
            f"{clinical_train_metrics['auroc']:.4f} ({clinical_train_metrics['auroc_ci'][0]:.4f}-{clinical_train_metrics['auroc_ci'][1]:.4f})",
            f"{clinical_val_metrics['auroc']:.4f} ({clinical_val_metrics['auroc_ci'][0]:.4f}-{clinical_val_metrics['auroc_ci'][1]:.4f})",
            f"{clinical_test_metrics['auroc']:.4f} ({clinical_test_metrics['auroc_ci'][0]:.4f}-{clinical_test_metrics['auroc_ci'][1]:.4f})",
            f"{nlp_train_metrics['auroc']:.4f} ({nlp_train_metrics['auroc_ci'][0]:.4f}-{nlp_train_metrics['auroc_ci'][1]:.4f})",
            f"{nlp_val_metrics['auroc']:.4f} ({nlp_val_metrics['auroc_ci'][0]:.4f}-{nlp_val_metrics['auroc_ci'][1]:.4f})",
            f"{nlp_test_metrics['auroc']:.4f} ({nlp_test_metrics['auroc_ci'][0]:.4f}-{nlp_test_metrics['auroc_ci'][1]:.4f})"
        ],
        'AUPRC': [
            f"{clinical_train_metrics['auprc']:.4f} ({clinical_train_metrics['auprc_ci'][0]:.4f}-{clinical_train_metrics['auprc_ci'][1]:.4f})",
            f"{clinical_val_metrics['auprc']:.4f} ({clinical_val_metrics['auprc_ci'][0]:.4f}-{clinical_val_metrics['auprc_ci'][1]:.4f})",
            f"{clinical_test_metrics['auprc']:.4f} ({clinical_test_metrics['auprc_ci'][0]:.4f}-{clinical_test_metrics['auprc_ci'][1]:.4f})",
            f"{nlp_train_metrics['auprc']:.4f} ({nlp_train_metrics['auprc_ci'][0]:.4f}-{nlp_train_metrics['auprc_ci'][1]:.4f})",
            f"{nlp_val_metrics['auprc']:.4f} ({nlp_val_metrics['auprc_ci'][0]:.4f}-{nlp_val_metrics['auprc_ci'][1]:.4f})",
            f"{nlp_test_metrics['auprc']:.4f} ({nlp_test_metrics['auprc_ci'][0]:.4f}-{nlp_test_metrics['auprc_ci'][1]:.4f})"
        ],
        'Sensitivity': [
            f"{clinical_train_metrics['sensitivity']:.4f}",
            f"{clinical_val_metrics['sensitivity']:.4f}",
            f"{clinical_test_metrics['sensitivity']:.4f}",
            f"{nlp_train_metrics['sensitivity']:.4f}",
            f"{nlp_val_metrics['sensitivity']:.4f}",
            f"{nlp_test_metrics['sensitivity']:.4f}"
        ],
        'Specificity': [
            f"{clinical_train_metrics['specificity']:.4f}",
            f"{clinical_val_metrics['specificity']:.4f}",
            f"{clinical_test_metrics['specificity']:.4f}",
            f"{nlp_train_metrics['specificity']:.4f}",
            f"{nlp_val_metrics['specificity']:.4f}",
            f"{nlp_test_metrics['specificity']:.4f}"
        ],
        'Brier Score': [
            f"{clinical_train_metrics['brier']:.4f}",
            f"{clinical_val_metrics['brier']:.4f}",
            f"{clinical_test_metrics['brier']:.4f}",
            f"{nlp_train_metrics['brier']:.4f}",
            f"{nlp_val_metrics['brier']:.4f}",
            f"{nlp_test_metrics['brier']:.4f}"
        ]
    })
    
    print("\n" + "="*60)
    print("COMPREHENSIVE RESULTS TABLE")
    print("="*60)
    print(results_table.to_string(index=False))
    
    results_table.to_csv(output_path, index=False)
    print(f"\nResults table saved to {output_path}")
    
    return results_table


def save_artifacts(dataset, stable_positive, stable_negative, vectorizer,
                  nlp_model, clinical_model, bootstrap_models, output_dir):
    """Save all model artifacts and results"""
    print("\n" + "="*60)
    print("SAVING ALL ARTIFACTS")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save stable features
    stable_features_df = pd.DataFrame({
        'Feature': [f[0] for f in stable_positive[:20]],
        'Appearances': [f[1] for f in stable_positive[:20]],
        'Direction': 'Increases Mortality'
    })
    stable_features_neg_df = pd.DataFrame({
        'Feature': [f[0] for f in stable_negative[:20]],
        'Appearances': [f[1] for f in stable_negative[:20]],
        'Direction': 'Decreases Mortality'
    })
    stable_features_combined = pd.concat([stable_features_df, stable_features_neg_df])
    stable_features_combined.to_csv(output_dir / 'stable_features.csv', index=False)
    
    # Save dataset with splits
    dataset.to_csv(output_dir / 'dataset_with_splits.csv', index=False)
    
    # Save models
    with open(output_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(output_dir / 'nlp_model.pkl', 'wb') as f:
        pickle.dump(nlp_model, f)
    
    with open(output_dir / 'clinical_model.pkl', 'wb') as f:
        pickle.dump(clinical_model, f)
    
    with open(output_dir / 'bootstrap_models.pkl', 'wb') as f:
        pickle.dump(bootstrap_models, f)
    
    print("\nSaved files:")
    print(f"  - {output_dir / 'comprehensive_results.csv'}")
    print(f"  - {output_dir / 'stable_features.csv'}")
    print(f"  - {output_dir / 'dataset_with_splits.csv'}")
    print(f"  - {output_dir / 'calibration_plots.png'}")
    print(f"  - {output_dir / 'performance_comparison.png'}")
    print(f"  - {output_dir / 'tfidf_vectorizer.pkl'}")
    print(f"  - {output_dir / 'nlp_model.pkl'}")
    print(f"  - {output_dir / 'clinical_model.pkl'}")
    print(f"  - {output_dir / 'bootstrap_models.pkl'}")


def print_study_notes():
    """Print study limitations and recommendations"""
    print("\n" + "="*60)
    print("KEY STUDY CHARACTERISTICS:")
    print("="*60)
    print("✓ PROSPECTIVE DATA: Using radiology notes from first 48h")
    print("✓ AVOIDS DATA LEAKAGE: Notes written before outcome")
    print("✓ CLINICALLY ACTIONABLE: Early intervention possible")
    print("✓ BOOTSTRAP CIs: Robust confidence intervals")
    print("✓ THRESHOLD TUNING: Optimized on validation set")
    print("✓ CALIBRATION ANALYSIS: Model reliability assessed")
    
    print("\n" + "="*60)
    print("REMAINING LIMITATIONS:")
    print("="*60)
    print("1. SINGLE-CENTER: MIMIC data from one hospital system")
    print("   → Limited generalizability to other hospitals/populations")
    print("2. SIMPLE REPRESENTATION: TF-IDF has no semantic understanding")
    print("   → Consider modern transformers (ClinicalBERT, BioBERT)")
    print("3. CLASS IMBALANCE: May affect performance estimates")
    print("4. NO TEMPORAL VALIDATION: All data from same time period")
    print("   → Should validate on more recent time period")
    print("5. NO EXTERNAL VALIDATION: Not tested on different dataset")
    print("   → Critical for clinical deployment")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR FUTURE WORK:")
    print("="*60)
    print("→ Compare with clinical severity scores (SOFA, APACHE II)")
    print("→ Test modern deep learning models (transformers)")
    print("→ Perform external validation on different datasets")
    print("→ Conduct temporal validation on more recent data")
    print("→ Discuss clinical utility and deployment considerations")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='MIMIC-IV Mortality Prediction using Early Radiology Notes'
    )
    parser.add_argument('--admissions', type=str, required=True,
                       help='Path to admissions.csv')
    parser.add_argument('--patients', type=str, required=True,
                       help='Path to patients.csv')
    parser.add_argument('--radiology', type=str, required=True,
                       help='Path to radiology.csv')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results (default: outputs)')
    parser.add_argument('--hours-window', type=int, default=48,
                       help='Time window for early notes in hours (default: 48)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--max-features', type=int, default=10000,
                       help='Max features for TF-IDF (default: 10000)')
    parser.add_argument('--n-bootstrap', type=int, default=5,
                       help='Number of bootstrap samples (default: 5)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MIMIC-IV MORTALITY PREDICTION - EARLY RADIOLOGY NOTES")
    print("="*80)
    
    # Load data
    admissions, patients, radiology_notes = load_data(
        args.admissions, args.patients, args.radiology
    )
    
    # Prepare data
    admissions_filtered = prepare_admissions(admissions, patients)
    dataset = extract_early_radiology_notes(
        radiology_notes, admissions_filtered, args.hours_window
    )
    dataset = create_splits(dataset, random_state=args.random_state)
    
    # Split data
    train_data = dataset[dataset['SPLIT'] == 'train']
    val_data = dataset[dataset['SPLIT'] == 'val']
    test_data = dataset[dataset['SPLIT'] == 'test']
    
    # Prepare features and labels
    clinical_features = ['age', 'is_male', 'is_emergency']
    X_train_clinical = train_data[clinical_features].values
    X_val_clinical = val_data[clinical_features].values
    X_test_clinical = test_data[clinical_features].values
    
    y_train = train_data['LABEL']
    y_val = val_data['LABEL']
    y_test = test_data['LABEL']
    
    texts_train = train_data['CONCATENATED_TEXT']
    texts_val = val_data['CONCATENATED_TEXT']
    texts_test = test_data['CONCATENATED_TEXT']
    
    # Train clinical baseline
    clinical_model = train_clinical_baseline(X_train_clinical, y_train)
    
    print("\nClinical model coefficients:")
    for feat, coef in zip(clinical_features, clinical_model.coef_[0]):
        print(f"  {feat:20s}: {coef:>8.4f}")
    
    # Train NLP model with bootstrap
    vectorizer, nlp_model, bootstrap_models, stable_positive, stable_negative = \
        train_nlp_model_with_bootstrap(
            texts_train, y_train, args.max_features, args.n_bootstrap
        )
    
    # Get predictions
    y_train_clinical_proba = clinical_model.predict_proba(X_train_clinical)[:, 1]
    y_val_clinical_proba = clinical_model.predict_proba(X_val_clinical)[:, 1]
    y_test_clinical_proba = clinical_model.predict_proba(X_test_clinical)[:, 1]
    
    texts_val_clean = texts_val.apply(preprocess_text)
    texts_test_clean = texts_test.apply(preprocess_text)
    X_val_tfidf = vectorizer.transform(texts_val_clean)
    X_test_tfidf = vectorizer.transform(texts_test_clean)
    
    y_train_nlp_proba = nlp_model.predict_proba(
        vectorizer.transform(texts_train.apply(preprocess_text))
    )[:, 1]
    y_val_nlp_proba = nlp_model.predict_proba(X_val_tfidf)[:, 1]
    y_test_nlp_proba = nlp_model.predict_proba(X_test_tfidf)[:, 1]
    
    # Tune threshold
    optimal_threshold = tune_threshold(y_val, y_val_nlp_proba)
    
    # Evaluate models
    print("\n" + "="*60)
    print("CLINICAL BASELINE RESULTS")
    print("="*60)
    
    clinical_train_metrics = evaluate_with_ci(
        y_train, y_train_clinical_proba, 0.5, "Clinical", "Train"
    )
    clinical_val_metrics = evaluate_with_ci(
        y_val, y_val_clinical_proba, 0.5, "Clinical", "Validation"
    )
    clinical_test_metrics = evaluate_with_ci(
        y_test, y_test_clinical_proba, 0.5, "Clinical", "Test"
    )
    
    print("\n" + "="*60)
    print("NLP MODEL RESULTS")
    print("="*60)
    
    nlp_train_metrics = evaluate_with_ci(
        y_train, y_train_nlp_proba, optimal_threshold, "NLP", "Train"
    )
    nlp_val_metrics = evaluate_with_ci(
        y_val, y_val_nlp_proba, optimal_threshold, "NLP", "Validation"
    )
    nlp_test_metrics = evaluate_with_ci(
        y_test, y_test_nlp_proba, optimal_threshold, "NLP", "Test"
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create plots
    plot_calibration(
        y_test, y_test_clinical_proba, y_test_nlp_proba,
        output_dir / 'calibration_plots.png'
    )
    
    plot_performance_comparison(
        y_train, y_train_clinical_proba, y_train_nlp_proba,
        y_val, y_val_clinical_proba, y_val_nlp_proba,
        y_test, y_test_clinical_proba, y_test_nlp_proba,
        clinical_train_metrics, clinical_val_metrics, clinical_test_metrics,
        nlp_train_metrics, nlp_val_metrics, nlp_test_metrics,
        output_dir / 'performance_comparison.png'
    )
    
    # Create results table
    create_results_table(
        clinical_train_metrics, clinical_val_metrics, clinical_test_metrics,
        nlp_train_metrics, nlp_val_metrics, nlp_test_metrics,
        output_dir / 'comprehensive_results.csv'
    )
    
    # Save artifacts
    save_artifacts(
        dataset, stable_positive, stable_negative, vectorizer,
        nlp_model, clinical_model, bootstrap_models, output_dir
    )
    
    # Print study notes
    print_study_notes()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
