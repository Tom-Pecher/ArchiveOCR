
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from src.utils import config, ProfessionalDigitClassifier, set_random_seeds
from src.train import load_datasets, train_epoch, TrainingMetrics
from src.eval import evaluate_model, create_professional_report
import numpy as np
import pandas as pd

# ================================================================
# REPRODUCIBILITY AND DEVICE CONFIGURATION
# ================================================================
set_random_seeds(42)

# Configure device with CUDA optimization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ================================================================
# MAIN TRAINING AND EVALUATION PIPELINE
# ================================================================
def main():
    """Main training and evaluation pipeline."""

    print("="*80)
    print("PROFESSIONAL OCR RESEARCH: MNIST DIGIT CLASSIFICATION")
    print("="*80)

    # ============ DATA PREPARATION ============
    print("\n1. Loading and preparing datasets...")
    train_loader, test_loader = load_datasets()

    # ============ MODEL INITIALIZATION ============
    print("\n2. Initializing model architecture...")
    model = ProfessionalDigitClassifier(num_classes=10).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ============ TRAINING SETUP ============
    print("\n3. Setting up training infrastructure...")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=config.LEARNING_RATE * 0.01
    )

    # Metrics tracking
    train_metrics = TrainingMetrics()
    test_metrics = TrainingMetrics()

    # ============ TRAINING LOOP ============
    print(f"\n4. Training for {config.EPOCHS} epochs...")

    best_test_acc = 0.0
    best_model_state = None

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, epoch
        )

        # Evaluate on test set
        test_loss, test_acc, _, _, _ = evaluate_model(
            model, test_loader, criterion, device, "Test"
        )

        # Update metrics
        current_lr = scheduler.get_last_lr()[0]
        train_metrics.update(train_loss, train_acc, current_lr)
        test_metrics.update(test_loss, test_acc, current_lr)

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            print(f"New best model! Test accuracy: {test_acc:.2f}%")

    # ============ FINAL EVALUATION ============
    print("\n5. Final evaluation with best model...")

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Generate comprehensive performance report
    final_metrics, confusion_mat, final_loss, final_acc = create_professional_report(
        model, test_loader, device
    )

    # ============ RESULTS VISUALIZATION ============
    print("\n6. Generating training visualizations...")

    # Create directories for plots if they don't exist
    os.makedirs('./data/plots', exist_ok=True)

    plt.figure(figsize=(15, 5))

    # Loss curves
    plt.subplot(1, 3, 1)
    epochs_range = range(1, len(train_metrics.losses) + 1)
    plt.plot(epochs_range, train_metrics.losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_range, test_metrics.losses, 'r-', label='Test Loss', linewidth=2)
    plt.title('Loss Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./data/plots/loss_curves.png')

    # Accuracy curves
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_metrics.accuracies, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, test_metrics.accuracies, 'r-', label='Test Accuracy', linewidth=2)
    plt.title('Accuracy Curves', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./data/plots/accuracy_curves.png')

    # Learning rate schedule
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_metrics.learning_rates, 'g-', linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('./data/plots/learning_rate.png')

    plt.tight_layout()
    plt.close() # Close the plot to free up memory

    # ============ MODEL PERSISTENCE ============
    print("\n7. Saving trained model...")

    # Save complete model information
    model_save_path = 'digit_classifier.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'final_test_accuracy': final_acc,
        'final_test_loss': final_loss,
        'epoch': config.EPOCHS,
        'model_architecture': str(model)
    }, model_save_path)

    print(f"Model saved to: {model_save_path}")

    # ============ SUMMARY REPORT ============
    print("\n" + "="*80)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Best Test Accuracy: {best_test_acc:.4f}%")
    print(f"Final Test Accuracy: {final_acc:.4f}%")
    print(f"Final Test Loss: {final_loss:.6f}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Training Time: {config.EPOCHS} epochs")
    print("="*80)

    # Save summary report to data/reports
    os.makedirs('./data/reports', exist_ok=True)
    report_path = './data/reports/final_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FINAL PERFORMANCE SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Best Test Accuracy: {best_test_acc:.4f}%\n")
        f.write(f"Final Test Accuracy: {final_acc:.4f}%\n")
        f.write(f"Final Test Loss: {final_loss:.6f}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Training Time: {config.EPOCHS} epochs\n")
        f.write("="*80 + "\n")

        f.write("\n" + "="*120 + "\n")
        f.write("COMPREHENSIVE PERFORMANCE METRICS\n")
        f.write("="*120 + "\n")
        f.write(final_metrics.to_string(index=False) + "\n")

        f.write("\n" + "="*60 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("="*60 + "\n")
        f.write("Rows: True Labels | Columns: Predicted Labels\n")
        f.write(np.array2string(confusion_mat, separator=', ') + "\n")

    print(f"\nSummary report saved to: {report_path}")

# ================================================================
# EXECUTION ENTRY POINT
# ================================================================
if __name__ == "__main__":
    main()
