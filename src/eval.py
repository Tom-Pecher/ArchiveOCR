import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from src.utils import config

def evaluate_model(model, test_loader, criterion, device, phase="Test"):
    """
    Evaluate the model on test/validation data.

    Returns:
        avg_loss: Average loss
        accuracy: Classification accuracy
        all_preds: All predictions for detailed analysis
        all_targets: All true labels
        all_probs: All prediction probabilities
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            outputs = model(data)
            test_loss += criterion(outputs, target).item()

            # Get predictions and probabilities
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Store for detailed analysis
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    print(f'{phase} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets), np.array(all_probs)

# ================================================================
# COMPREHENSIVE OCR PERFORMANCE ANALYSIS
# ================================================================
def calculate_comprehensive_metrics(y_true, y_pred, y_probs, class_names=None):
    """
    Calculate comprehensive performance metrics for OCR evaluation.

    This follows standard practices in OCR research literature.
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Initialize metrics storage
    metrics_data = []

    # Calculate per-class metrics
    for class_idx in range(len(class_names)):
        # Confusion matrix components
        tp = cm[class_idx, class_idx]
        fp = cm[:, class_idx].sum() - tp
        fn = cm[class_idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        # Core classification metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Also known as sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Additional OCR-relevant metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        balanced_accuracy = (recall + specificity) / 2
        error_rate = (fp + fn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        # Confidence analysis for correctly classified samples
        correct_mask = (y_true == class_idx) & (y_pred == class_idx)
        avg_confidence = y_probs[correct_mask, class_idx].mean() if correct_mask.sum() > 0 else 0.0

        # Support (number of true instances)
        support = (y_true == class_idx).sum()

        # Matthews Correlation Coefficient (MCC) - robust for imbalanced classes
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

        metrics_data.append({
            'Class': class_names[class_idx],
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Specificity': specificity,
            'Accuracy': accuracy,
            'Balanced Acc': balanced_accuracy,
            'MCC': mcc,
            'Error Rate': error_rate,
            'Avg Confidence': avg_confidence,
            'Support': support,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn
        })

    # Create DataFrame
    df = pd.DataFrame(metrics_data)

    # Calculate aggregate metrics
    total_support = df['Support'].sum()

    # Macro averages (unweighted)
    macro_metrics = {
        'Class': 'Macro Avg',
        'Precision': df['Precision'].mean(),
        'Recall': df['Recall'].mean(),
        'F1-Score': df['F1-Score'].mean(),
        'Specificity': df['Specificity'].mean(),
        'Accuracy': df['Accuracy'].mean(),
        'Balanced Acc': df['Balanced Acc'].mean(),
        'MCC': df['MCC'].mean(),
        'Error Rate': df['Error Rate'].mean(),
        'Avg Confidence': df['Avg Confidence'].mean(),
        'Support': total_support,
        'TP': df['TP'].sum(),
        'FP': df['FP'].sum(),
        'FN': df['FN'].sum(),
        'TN': df['TN'].sum()
    }

    # Weighted averages (by support)
    weights = df['Support'] / total_support
    weighted_metrics = {
        'Class': 'Weighted Avg',
        'Precision': (df['Precision'] * weights).sum(),
        'Recall': (df['Recall'] * weights).sum(),
        'F1-Score': (df['F1-Score'] * weights).sum(),
        'Specificity': (df['Specificity'] * weights).sum(),
        'Accuracy': (df['Accuracy'] * weights).sum(),
        'Balanced Acc': (df['Balanced Acc'] * weights).sum(),
        'MCC': (df['MCC'] * weights).sum(),
        'Error Rate': (df['Error Rate'] * weights).sum(),
        'Avg Confidence': (df['Avg Confidence'] * weights).sum(),
        'Support': total_support,
        'TP': df['TP'].sum(),
        'FP': df['FP'].sum(),
        'FN': df['FN'].sum(),
        'TN': df['TN'].sum()
    }

    # Add aggregate rows to dataframe
    df = pd.concat([df, pd.DataFrame([macro_metrics, weighted_metrics])], ignore_index=True)

    return df, cm

def analyze_error_patterns(y_true, y_pred, cm, class_names=None):
    """
    Analyze error patterns specific to OCR applications.
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    print("\n" + "="*80)
    print("ERROR PATTERN ANALYSIS FOR OCR")
    print("="*80)

    # Overall system performance
    overall_accuracy = (y_true == y_pred).mean()
    character_error_rate = 1 - overall_accuracy

    print(f"Character Recognition Accuracy: {overall_accuracy:.4f}")
    print(f"Character Error Rate (CER): {character_error_rate:.4f}")

    # Find most frequent misclassifications
    print(f"\nTop 10 Most Frequent Misclassifications:")
    print("-" * 50)

    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                error_rate = cm[i, j] / cm[i, :].sum()
                confusion_pairs.append((class_names[i], class_names[j], cm[i, j], error_rate))

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    for true_class, pred_class, count, rate in confusion_pairs[:10]:
        print(f"  {true_class} → {pred_class}: {count:4d} errors ({rate:.3f} of all {true_class}s)")

    # Identify problematic digit pairs (high mutual confusion)
    print(f"\nMutually Confused Digit Pairs:")
    print("-" * 40)

    mutual_confusions = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            mutual_errors = cm[i, j] + cm[j, i]
            if mutual_errors > 5:  # Threshold for significance
                mutual_confusions.append((class_names[i], class_names[j], mutual_errors))

    mutual_confusions.sort(key=lambda x: x[2], reverse=True)

    for class1, class2, errors in mutual_confusions[:5]:
        print(f"  {class1} ↔ {class2}: {errors} mutual errors")

    # Per-class error analysis
    print(f"\nPer-Class Error Analysis:")
    print("-" * 30)

    class_errors = []
    for i in range(len(class_names)):
        total_samples = cm[i, :].sum()
        correct = cm[i, i]
        errors = total_samples - correct
        error_rate = errors / total_samples if total_samples > 0 else 0
        class_errors.append((class_names[i], errors, error_rate, total_samples))

    class_errors.sort(key=lambda x: x[2], reverse=True)

    print("Class | Errors | Error Rate | Total Samples")
    print("-" * 45)
    for class_name, errors, error_rate, total in class_errors:
        print(f"  {class_name}   |   {errors:3d}  |    {error_rate:.3f}   |     {total:4d}")

def create_professional_report(model, test_loader, device):
    """
    Generate a comprehensive performance report following OCR research standards.
    """
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE OCR PERFORMANCE REPORT")
    print("="*80)

    # Evaluate model and get detailed predictions
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_pred, y_true, y_probs = evaluate_model(
        model, test_loader, criterion, device, "Final Test"
    )

    # Calculate comprehensive metrics
    metrics_df, confusion_matrix = calculate_comprehensive_metrics(y_true, y_pred, y_probs)

    # Display results with professional formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print("\n" + "="*120)
    print("COMPREHENSIVE PERFORMANCE METRICS")
    print("="*120)
    print(metrics_df.to_string(index=False))

    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print("Rows: True Labels | Columns: Predicted Labels")
    print(confusion_matrix)

    # Detailed error analysis
    analyze_error_patterns(y_true, y_pred, confusion_matrix)

    return metrics_df, confusion_matrix, test_loss, test_acc