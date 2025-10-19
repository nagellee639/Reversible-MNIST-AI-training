import numpy as np
import os
import sys
import time
import tensorflow as tf
# No random needed as GPU handles deterministic tie-breaking internally now

# --- Import our fast C++ GPU module ---
try:
    import fast_network_gpu as fast_network
except ImportError:
    print("Error: The 'fast_network_gpu' module was not found.")
    print("Please compile the C++ GPU code first by running: python setup_gpu.py install")
    sys.exit(1)

# --- Constants ---
# Use the GPU-specific names
CHAMPION_FILE = "champion_chromosome_gpu.npy"
# BACKUP_DIR is not needed as we require the champion file

def load_mnist_data():
    """Loads and binarizes both training and test data."""
    print("Loading and preparing MNIST datasets...")
    # We still need a tiny bit of training data as placeholder for init_gpu
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Take only a small slice of training data as dummy input
    x_train_flat = (x_train[:10] > 127).astype(np.uint8).reshape(10, -1)
    y_train_dummy = y_train[:10]
    # Prepare the full test data
    x_test_flat = (x_test > 127).astype(np.uint8).reshape(x_test.shape[0], -1)
    print("Datasets loaded.")
    return (x_train_flat, y_train_dummy), (x_test_flat, y_test)

def load_champion():
    """Loads the champion chromosome directly."""
    if not os.path.exists(CHAMPION_FILE):
        print(f"Error: Champion file '{CHAMPION_FILE}' not found.")
        print("Please run train_gpu.py to generate a champion first.")
        sys.exit(1)

    print(f"Loading champion from '{CHAMPION_FILE}'...")
    try:
        champion = np.load(CHAMPION_FILE)
        # Basic validation
        if champion.shape == (fast_network.CHROMOSOME_LENGTH,) and champion.dtype == np.uint8:
            return champion
        else:
            print(f"Error: Champion file '{CHAMPION_FILE}' has incorrect shape or dtype.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading champion file: {e}")
        sys.exit(1)

# --- Removed score_output function (logic is inside evaluate_champion_gpu) ---
# --- Removed find_or_load_champion function (replaced by simpler load_champion) ---

# --- Main Evaluation Logic ---
if __name__ == "__main__":
    # Load dummy train data and real test data
    (dummy_train_images, dummy_train_labels), (test_images, test_labels) = load_mnist_data()

    champion_chromosome = load_champion()

    print("\nInitializing GPU and transferring test dataset...")
    # --- Initialize GPU ---
    # Pass dummy data for evolution set, and TEST data for validation set
    fast_network.init_gpu(
        dummy_train_images.flatten().tolist(), dummy_train_labels.tolist(),
        test_images.flatten().tolist(), test_labels.tolist()
    )

    print(f"\n--- Starting GPU Confusion Matrix Analysis on {len(test_images)} Test Images ---")

    start_time = time.time()
    # --- Call the GPU function ---
    # This runs the optimized kernel over all test images (loaded as validation data)
    # and returns the completed confusion matrix.
    confusion_matrix_flat = fast_network.evaluate_champion_gpu(champion_chromosome.tolist())
    end_time = time.time()

    print(f"\nGPU Analysis completed in {end_time - start_time:.2f} seconds.")

    # Reshape the flat matrix returned by C++
    confusion_matrix = np.array(confusion_matrix_flat, dtype=int).reshape((10, 10))

    # --- Print Results (Identical formatting to your original script) ---
    print("\n" + "="*70)
    print("                    UNBIASED CONFUSION MATRIX (GPU)")
    print("="*70)
    print("       Predicted ->")
    print("True |    0    1    2    3    4    5    6    7    8    9")
    print("-----+-------------------------------------------------------")
    for i in range(10):
        row_str = f"  {i}  | "
        for j in range(10):
            row_str += f"{confusion_matrix[i, j]:>4} "
        print(row_str)
    print("="*70)

    # Calculate overall accuracy
    total_correct = np.trace(confusion_matrix)
    total_examples = np.sum(confusion_matrix)
    overall_accuracy = total_correct / total_examples if total_examples > 0 else 0

    print("\n" + "="*70)
    print("                       UNBIASED PER-DIGIT ANALYSIS (GPU)")
    print("="*70)

    # Display overall accuracy prominently
    print(f"\n>>> OVERALL ACCURACY: {total_correct}/{total_examples} ({overall_accuracy:.2%}) <<<\n")

    for i in range(10):
        total_in_class = np.sum(confusion_matrix[i])
        correct_in_class = confusion_matrix[i, i]
        accuracy = correct_in_class / total_in_class if total_in_class > 0 else 0.0 # Avoid division by zero

        print(f"\n--- True Digit: {i} ({total_in_class} examples) ---")
        print(f"  Accuracy: {correct_in_class} / {total_in_class}  ({accuracy:.2%})")

        # Get predictions sorted by count for this true class
        predictions = sorted(enumerate(confusion_matrix[i]), key=lambda x: x[1], reverse=True)

        print("  Most Common Guesses:")
        # Show top 5 guesses or fewer if less than 5 non-zero guesses exist
        count_shown = 0
        for digit, count in predictions:
            if count > 0 and count_shown < 5:
                marker = "(Correct)" if digit == i else ""
                print(f"    - Guessed '{digit}': {count} times {marker}")
                count_shown += 1
        if count_shown == 0:
            print("    - (No predictions recorded for this digit)")


    print("\n" + "="*70)
    print(f"FINAL OVERALL ACCURACY: {total_correct}/{total_examples} ({overall_accuracy:.2%})")
    print("="*70)
