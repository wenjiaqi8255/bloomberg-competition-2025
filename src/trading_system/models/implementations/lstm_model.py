"""
LSTM Regression Model

This module implements a Long Short-Term Memory (LSTM) neural network for
time series prediction in trading. LSTMs are particularly effective for:
- Sequential patterns
- Long-term dependencies
- Temporal dynamics
- Multi-step forecasting

Key Features:
- Recurrent neural architecture
- Sequence modeling
- Dropout regularization
- Early stopping support
"""

import logging
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    Dataset = object  # Dummy base class
    DataLoader = None

from ..base.base_model import BaseModel, ModelStatus, ModelMetadata

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Features array of shape (n_samples, sequence_length, n_features)
            y: Targets array of shape (n_samples,)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TimeSeriesDataset")
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if TORCH_AVAILABLE:
    _base_class = nn.Module
else:
    _base_class = object

class LSTMNetwork(_base_class):
    """LSTM neural network architecture."""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTMNetwork")
        super(LSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        dropped = self.dropout(last_output)
        
        # Output layer
        output = self.fc(dropped)
        
        return output.squeeze(-1)


class LSTMModel(BaseModel):
    """
    LSTM Regression Model for trading signal prediction.
    
    This model uses recurrent neural networks to capture temporal patterns
    in sequential financial data.
    
    Example Usage:
        # Create model
        model = LSTMModel(config={
            'sequence_length': 20,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        })
        
        # Train (X should be sequences)
        model.fit(X_train, y_train, X_val, y_val)
        
        # Predict
        predictions = model.predict(X_test)
    """
    
    def __init__(self, model_type: str = "lstm", config: Optional[Dict[str, Any]] = None):
        """
        Initialize LSTM model.
        
        Args:
            model_type: Model identifier
            config: Configuration dictionary:
                - sequence_length: Length of input sequences (default: 20)
                - hidden_size: Number of LSTM hidden units (default: 64)
                - num_layers: Number of LSTM layers (default: 2)
                - dropout: Dropout probability (default: 0.2)
                - bidirectional: Use bidirectional LSTM (default: False)
                - learning_rate: Adam optimizer learning rate (default: 0.001)
                - batch_size: Training batch size (default: 32)
                - epochs: Number of training epochs (default: 100)
                - early_stopping_patience: Early stopping patience (default: 10)
                - device: 'cuda' or 'cpu' (default: auto-detect)
        """
        super().__init__(model_type, config)
        
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Please install it with: "
                "pip install torch"
            )
        
        # Extract hyperparameters
        self.sequence_length = self.config.get('sequence_length', 20)
        self.hidden_size = self.config.get('hidden_size', 64)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.2)
        self.bidirectional = self.config.get('bidirectional', False)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 100)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        
        # Device configuration
        device_config = self.config.get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        # Model will be initialized during training
        self._model = None
        self._input_size = None
        self._feature_names = None
        
        # Normalization parameters (for input scaling)
        self._scaler_mean = None
        self._scaler_std = None
        
        logger.info(f"Initialized LSTMModel on device: {self.device}")
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert tabular data to sequences.
        
        Args:
            X: Feature DataFrame
            y: Target Series
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_values = X.values
        y_values = y.values
        
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X_values) - self.sequence_length + 1):
            X_sequences.append(X_values[i:i + self.sequence_length])
            y_sequences.append(y_values[i + self.sequence_length - 1])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _normalize_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize input data using z-score normalization.
        
        Args:
            X: Input array
            fit: Whether to fit normalization parameters
        
        Returns:
            Normalized array
        """
        if fit:
            self._scaler_mean = np.mean(X, axis=(0, 1), keepdims=True)
            self._scaler_std = np.std(X, axis=(0, 1), keepdims=True) + 1e-8
        
        if self._scaler_mean is not None and self._scaler_std is not None:
            return (X - self._scaler_mean) / self._scaler_std
        else:
            return X
    
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'LSTMModel':
        """
        Train the LSTM model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
        
        Returns:
            Self for method chaining
        """
        try:
            # Validate input data
            self.validate_data(X, y)
            
            # Clean data
            aligned_data = pd.concat([y, X], axis=1).dropna()
            if len(aligned_data) == 0:
                raise ValueError("No valid data points after alignment")
            
            y_clean = aligned_data.iloc[:, 0]
            X_clean = aligned_data.iloc[:, 1:]
            
            # Store feature names
            self._feature_names = list(X_clean.columns)
            self._input_size = len(self._feature_names)
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X_clean, y_clean)
            
            if len(X_seq) == 0:
                raise ValueError(f"Not enough data for sequence length {self.sequence_length}")
            
            # Normalize
            X_seq = self._normalize_data(X_seq, fit=True)
            
            # Create dataset and dataloader
            train_dataset = TimeSeriesDataset(X_seq, y_seq)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # Prepare validation data if available
            val_loader = None
            if X_val is not None and y_val is not None:
                val_aligned = pd.concat([y_val, X_val[self._feature_names]], axis=1).dropna()
                if len(val_aligned) > 0:
                    y_val_clean = val_aligned.iloc[:, 0]
                    X_val_clean = val_aligned.iloc[:, 1:]
                    
                    X_val_seq, y_val_seq = self._create_sequences(X_val_clean, y_val_clean)
                    if len(X_val_seq) > 0:
                        X_val_seq = self._normalize_data(X_val_seq, fit=False)
                        val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)
                        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
            
            # Initialize model
            self._model = LSTMNetwork(
                input_size=self._input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            for epoch in range(self.epochs):
                # Training
                self._model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self._model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation
                if val_loader:
                    self._model.eval()
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X = batch_X.to(self.device)
                            batch_y = batch_y.to(self.device)
                            outputs = self._model(batch_X)
                            loss = criterion(outputs, batch_y)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = self._model.state_dict().copy()
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
                    
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch + 1}/{self.epochs} - "
                                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch + 1}/{self.epochs} - "
                                  f"Train Loss: {train_loss:.6f}")
            
            # Load best model if validation was used
            if best_model_state is not None:
                self._model.load_state_dict(best_model_state)
            
            # Update status and metadata
            self.status = ModelStatus.TRAINED
            self.metadata.training_samples = len(y_seq)
            self.metadata.features = self._feature_names
            
            self.metadata.hyperparameters.update({
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
                'learning_rate': self.learning_rate,
                'best_val_loss': best_val_loss if val_loader else None
            })
            
            logger.info(f"Successfully trained LSTMModel on {len(y_seq)} sequences")
            
            return self
        
        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"Failed to train LSTMModel: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame (must have at least sequence_length rows)
        
        Returns:
            Array of predictions
        """
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Validate input data
            self.validate_data(X)
            
            # Ensure correct features
            if self._feature_names:
                missing_features = set(self._feature_names) - set(X.columns)
                if missing_features:
                    raise ValueError(f"Missing features: {missing_features}")
                X_pred = X[self._feature_names]
            else:
                X_pred = X
            
            # For prediction, we take the last sequence_length rows
            if len(X_pred) < self.sequence_length:
                raise ValueError(
                    f"Need at least {self.sequence_length} rows for prediction, "
                    f"got {len(X_pred)}"
                )
            
            # Create sequence
            X_values = X_pred.values[-self.sequence_length:]
            X_seq = X_values.reshape(1, self.sequence_length, -1)
            
            # Normalize
            X_seq = self._normalize_data(X_seq, fit=False)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            
            # Make prediction
            self._model.eval()
            with torch.no_grad():
                prediction = self._model(X_tensor)
            
            result = prediction.cpu().numpy()
            
            logger.debug("Made LSTM prediction")
            return result
        
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': self.model_type,
            'status': self.status,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'training_samples': self.metadata.training_samples,
            'n_features': self._input_size,
            'device': str(self.device)
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the LSTM model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        if self._model is not None:
            model_path = path / "lstm_model.pt"
            torch.save(self._model.state_dict(), model_path)
        
        # Save additional state
        state_dict = {
            'feature_names': self._feature_names,
            'input_size': self._input_size,
            'scaler_mean': self._scaler_mean,
            'scaler_std': self._scaler_std,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional
        }
        
        state_path = path / "model_state.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(state_dict, f)
        
        # Save metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        # Save config
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"LSTMModel saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LSTMModel':
        """Load an LSTM model from disk."""
        path = Path(path)
        
        if not path.exists():
            raise ValueError(f"Model path does not exist: {path}")
        
        # Load config
        config_path = path / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Create instance
        instance = cls(config=config)
        
        # Load state
        state_path = path / "model_state.pkl"
        with open(state_path, 'rb') as f:
            state_dict = pickle.load(f)
        
        instance._feature_names = state_dict['feature_names']
        instance._input_size = state_dict['input_size']
        instance._scaler_mean = state_dict['scaler_mean']
        instance._scaler_std = state_dict['scaler_std']
        
        # Recreate model architecture
        instance._model = LSTMNetwork(
            input_size=instance._input_size,
            hidden_size=instance.hidden_size,
            num_layers=instance.num_layers,
            dropout=instance.dropout,
            bidirectional=instance.bidirectional
        ).to(instance.device)
        
        # Load model weights
        model_path = path / "lstm_model.pt"
        instance._model.load_state_dict(torch.load(model_path, map_location=instance.device))
        
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            instance.metadata = ModelMetadata.from_dict(metadata_dict)
        
        instance.status = ModelStatus.TRAINED
        
        logger.info(f"LSTMModel loaded from {path}")
        return instance

