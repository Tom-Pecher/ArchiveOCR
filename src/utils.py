
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

# ================================================================
# REPRODUCIBILITY AND DEVICE CONFIGURATION
# ================================================================
def set_random_seeds(seed=42):
    """Set random seeds for reproducible results across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================================================================
# HYPERPARAMETER CONFIGURATION
# ================================================================
class Config:
    """Centralized configuration following research best practices."""
    # Model architecture parameters
    CONV1_CHANNELS = 32
    CONV2_CHANNELS = 64
    CONV3_CHANNELS = 128
    FC1_UNITS = 256
    FC2_UNITS = 128
    DROPOUT_CONV = 0.25
    DROPOUT_FC = 0.5

    # Training hyperparameters
    BATCH_SIZE = 64  # Smaller batch for better gradient estimates
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4  # L2 regularization
    EPOCHS = 5

    # Data preprocessing parameters
    NORMALIZE_MEAN = 0.1307  # MNIST dataset statistics
    NORMALIZE_STD = 0.3081

    # Evaluation parameters
    CONFIDENCE_THRESHOLD = 0.9
    PRINT_FREQ = 100

config = Config()

# ================================================================
# ADVANCED CNN ARCHITECTURE FOR OCR RESEARCH
# ================================================================
class ProfessionalDigitClassifier(nn.Module):
    """
    State-of-the-art CNN architecture for digit classification.

    Features:
    - Three convolutional layers with increasing depth
    - Batch normalization for training stability
    - Dropout for regularization
    - Residual-like connections for better gradient flow
    - Adaptive pooling for variable input sizes
    """

    def __init__(self, num_classes=10):
        super(ProfessionalDigitClassifier, self).__init__()

        # ============ CONVOLUTIONAL FEATURE EXTRACTOR ============
        # First convolutional block
        self.conv1 = nn.Conv2d(1, config.CONV1_CHANNELS, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(config.CONV1_CHANNELS)

        # Second convolutional block
        self.conv2 = nn.Conv2d(config.CONV1_CHANNELS, config.CONV2_CHANNELS, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(config.CONV2_CHANNELS)

        # Third convolutional block for deeper feature extraction
        self.conv3 = nn.Conv2d(config.CONV2_CHANNELS, config.CONV3_CHANNELS, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(config.CONV3_CHANNELS)

        # Pooling and regularization layers
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.dropout_conv = nn.Dropout2d(config.DROPOUT_CONV)

        # ============ CLASSIFICATION HEAD ============
        # Calculate the size after convolutions for first FC layer
        self.fc1 = nn.Linear(config.CONV3_CHANNELS, config.FC1_UNITS)
        self.fc2 = nn.Linear(config.FC1_UNITS, config.FC2_UNITS)
        self.fc3 = nn.Linear(config.FC2_UNITS, num_classes)

        self.dropout_fc = nn.Dropout(config.DROPOUT_FC)

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network."""
        # ============ FEATURE EXTRACTION ============
        # First convolutional block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)

        # Second convolutional block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)

        # Third convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)  # Global average pooling

        # ============ CLASSIFICATION ============
        # Flatten features
        x = torch.flatten(x, 1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)

        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)

        # Output layer (logits)
        x = self.fc3(x)

        return x
    