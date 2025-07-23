import matplotlib.pyplot as plt
import numpy as np
import os
from config import building_name_list

# Function to load loss data
def load_loss_data(building_name, model_name='lstm'):
    """
    Load train and test loss data from .npy files for a given building and model.
    Uses allow_pickle=True to handle object arrays safely.
    """
    train_loss_path = f'./results/{building_name}/lstm_train_loss.npy'
    test_loss_path = f'./results/{building_name}/lstm_test_loss.npy'
    
    # Check if files exist
    if not os.path.exists(train_loss_path):
        raise FileNotFoundError(f"Train loss file not found: {train_loss_path}")
    if not os.path.exists(test_loss_path):
        raise FileNotFoundError(f"Test loss file not found: {test_loss_path}")
    
    try:
        # Load data with allow_pickle=True to handle object arrays
        train_loss = np.load(train_loss_path, allow_pickle=True)
        test_loss = np.load(test_loss_path, allow_pickle=True)
        
        # Ensure data is numeric (convert if necessary)
        train_loss = np.array(train_loss, dtype=float)
        test_loss = np.array(test_loss, dtype=float)
        
        return train_loss, test_loss
    except Exception as e:
        raise ValueError(f"Error loading loss data for {building_name}: {str(e)}")

# Plotting function
def plot_train_test_loss(building_name, train_loss, test_loss):
    """
    Plot and save train vs test loss for a given building.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, test_loss, label='Test Loss', color='orange', linewidth=2)
    
    plt.title(f'Train vs Test Loss for LSTM Model - {building_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    os.makedirs(f'./plots/{building_name}', exist_ok=True)
    plt.savefig(f'./plots/{building_name}/lstm_train_test_loss.png')
    plt.close()

# Main execution
if __name__ == '__main__':
    for building_name in building_name_list:
        try:
            # Load loss data
            train_loss, test_loss = load_loss_data(building_name, model_name='lstm')
            # Plot the train vs test loss
            plot_train_test_loss(building_name, train_loss, test_loss)
            print(f"Successfully generated plot for {building_name}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error for {building_name}: {e}")