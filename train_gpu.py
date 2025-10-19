import numpy as np
import time
import os
import sys
import tensorflow as tf

# --- Import our fast C++ GPU module ---
try:
    import fast_network_gpu as fast_network
except ImportError:
    print("Error: The 'fast_network_gpu' module was not found.")
    print("Please compile the C++ GPU code first by running: python setup_gpu.py install")
    sys.exit(1)

# --- Genetic Algorithm Constants ---
POPULATION_SIZE = 1000
NUM_GENERATIONS = 80000 # Keep your large target
MINI_BATCH_SIZE = 1000
ELITISM_COUNT = 2
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.005
CHAMPION_REPLACEMENT_THRESHOLD = 0.005 # 0.5% improvement required

# --- NEW: Specify which digits to focus training on ---
# Example: Focus on digits 1, 7, and 0
TARGET_DIGITS = [0, 1, 2, 3, 4, 6, 7, 8]
# Example: Focus on just 0 and 1
# TARGET_DIGITS = [0, 1]
# Example: Focus on all digits (original behavior)
# TARGET_DIGITS = list(range(10))


# --- Generation threshold to enable dynamic weights ---
ENABLE_WEIGHTS_AFTER_GEN = 350 # Keep delay or set to 0 to enable immediately

# --- File System Constants ---
BACKUP_DIR = "population_backups_gpu"
CHAMPION_FILE = "champion_chromosome_gpu.npy"
BACKUP_INTERVAL = 5

# --- Network Constants ---
CHROMOSOME_LENGTH = fast_network.CHROMOSOME_LENGTH
IMAGE_SIZE = fast_network.IMAGE_SIZE
BITS_PER_TRANSFORMER = fast_network.BITS_PER_TRANSFORMER
NUM_TRANSFORMERS = fast_network.NUM_TRANSFORMERS


# --- Champion evaluation function (uses GPU) ---
def evaluate_champion_robust(chromosome):
    """
    Evaluate champion on the entire validation set using the GPU.
    Returns accuracy and per-class accuracies. (Accuracy is across ALL digits evaluated)
    """
    confusion_matrix_flat = fast_network.evaluate_champion_gpu(chromosome.tolist())
    confusion_matrix = np.array(confusion_matrix_flat).reshape((10, 10))
    total_correct = np.sum(np.diag(confusion_matrix))
    total_predictions = np.sum(confusion_matrix)
    # Handle potential division by zero if total_predictions is 0
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
    class_totals = np.sum(confusion_matrix, axis=1)
    # Calculate per-class accuracies, ensuring 0/0 results in 0.0 accuracy
    per_class_accuracies = np.divide(np.diag(confusion_matrix), class_totals,
                                     out=np.zeros_like(np.diag(confusion_matrix), dtype=float),
                                     where=class_totals!=0)
    return accuracy, per_class_accuracies, confusion_matrix

# --- Dynamic class weighting ---
def update_class_weights(per_class_accuracies, current_weights, generation, target_digits):
    """
    Update class weights based on current performance FOR TARGET DIGITS.
    Weights for non-target digits are effectively ignored (will be set to 0 later).
    """
    new_weights = np.zeros(10, dtype=float) # Initialize all weights to 0
    # Calculate weights only for target digits based on their accuracy
    target_accuracies = np.array([per_class_accuracies[i] for i in target_digits])
    target_raw_weights = 1.0 - target_accuracies
    target_clipped_weights = np.clip(target_raw_weights, 0.1, 0.9) # Avoid extremes

    # Place calculated weights into the full 10-element array
    for idx, digit in enumerate(target_digits):
        new_weights[digit] = target_clipped_weights[idx]

    # Smooth update using only the weights of target digits
    blend_factor = 0.7
    if current_weights is None:
        # On first update, just use the new weights (non-targets are already 0)
        return new_weights
    else:
        # Blend only the target digits' weights
        blended_weights = np.zeros(10, dtype=float)
        for digit in target_digits:
            blended_weights[digit] = blend_factor * new_weights[digit] + (1 - blend_factor) * current_weights[digit]
        # Non-target digits remain 0
        return blended_weights

# --- Backup/Load Functions ---
def save_backup(population, generation):
    """Saves the current population to a file."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    filename = os.path.join(BACKUP_DIR, f"population_gen_{generation}.npy")
    np.save(filename, population)
    print(f"  Backup saved to {filename}")

def load_from_backup():
    """Loads the latest population backup if one exists."""
    if not os.path.isdir(BACKUP_DIR): return None, 0
    backups = [f for f in os.listdir(BACKUP_DIR) if f.endswith('.npy')]
    if not backups: return None, 0
    try:
        # Find the backup with the highest generation number
        latest_backup = max(backups, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        generation = int(latest_backup.split('_')[-1].split('.')[0])
        print(f"Loading population from latest backup: {latest_backup}")
        population = np.load(os.path.join(BACKUP_DIR, latest_backup))
        # Add basic validation for shape and dtype
        if population.shape == (POPULATION_SIZE, CHROMOSOME_LENGTH) and population.dtype == np.uint8:
             return population, generation
        else:
             print(f"  Warning: Backup file {latest_backup} has incorrect shape ({population.shape}) or dtype ({population.dtype}). Expected ({POPULATION_SIZE}, {CHROMOSOME_LENGTH}) and uint8. Ignoring.")
             return None, 0
    except (ValueError, IndexError, FileNotFoundError) as e:
         # Catch potential errors during parsing or loading
         print(f"  Warning: Could not load or parse backup files ({e}). Starting fresh.")
         return None, 0


# --- Main Training Loop ---
if __name__ == "__main__":
    print("--- GPU-Accelerated Reversible Network Trainer ---")
    print(f"--- Focusing training on digits: {TARGET_DIGITS} ---")

    print("Loading and preparing MNIST dataset...")
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train_binary = (x_train > 127).astype(np.uint8)
    training_images = x_train_binary.reshape(x_train.shape[0], -1)
    training_labels = y_train
    total_samples = len(training_images)
    validation_size = min(10000, total_samples // 5) # 20%
    # np.random.seed(42) # Optional: for reproducible splits
    validation_indices = np.random.choice(total_samples, validation_size, replace=False)
    validation_mask = np.zeros(total_samples, dtype=bool)
    validation_mask[validation_indices] = True
    # Initial split
    evo_imgs_full = training_images[~validation_mask]
    evo_lbls_full = training_labels[~validation_mask]
    val_imgs_full = training_images[validation_mask]
    val_lbls_full = training_labels[validation_mask]

    # --- FILTER DATASETS based on TARGET_DIGITS ---
    print(f"Filtering datasets for target digits: {TARGET_DIGITS}...")
    target_set = set(TARGET_DIGITS)

    evo_filter_mask = np.isin(evo_lbls_full, TARGET_DIGITS)
    evolution_images = evo_imgs_full[evo_filter_mask]
    evolution_labels = evo_lbls_full[evo_filter_mask]

    val_filter_mask = np.isin(val_lbls_full, TARGET_DIGITS)
    validation_images = val_imgs_full[val_filter_mask]
    validation_labels = val_lbls_full[val_filter_mask]

    print(f"Filtered Training set: {len(evolution_images)} images")
    print(f"Filtered Validation set: {len(validation_images)} images")
    if len(evolution_images) == 0 or len(validation_images) == 0:
        print("Error: Filtering resulted in empty evolution or validation set. Check TARGET_DIGITS.")
        sys.exit(1)

    print("Transferring dataset to GPU memory...")
    # Pass the FILTERED datasets to the GPU
    fast_network.init_gpu(
        evolution_images.flatten().tolist(), evolution_labels.tolist(),
        validation_images.flatten().tolist(), validation_labels.tolist()
    )

    population, start_generation = load_from_backup()
    if population is None:
        print("No backup found. Initializing new random population.")
        population = np.random.randint(0, 2, size=(POPULATION_SIZE, CHROMOSOME_LENGTH), dtype=np.uint8)
        start_generation = 0

    current_champion = None
    current_champion_validation_accuracy = 0.0
    champion_generations_survived = 0
    # Initialize class_weights to None
    class_weights = None
    # Flag to track if weights have been enabled
    weights_enabled = False

    # --- Champion loading ---
    if os.path.exists(CHAMPION_FILE) and start_generation > 0:
        print(f"Loading champion from {CHAMPION_FILE}...")
        try:
            current_champion = np.load(CHAMPION_FILE)
            if current_champion.shape == (CHROMOSOME_LENGTH,) and current_champion.dtype == np.uint8:
                 # Evaluate loaded champion on the (filtered) validation set
                 acc, per_class, _ = evaluate_champion_robust(current_champion)
                 current_champion_validation_accuracy = acc
                 print(f"  Loaded champion accuracy (on filtered valid set): {acc*100:.2f}%")
                 # --- Initialize weights ONLY if past the threshold ---
                 if start_generation >= ENABLE_WEIGHTS_AFTER_GEN:
                     # Calculate weights based on performance on target digits
                     class_weights = update_class_weights(per_class, None, start_generation, TARGET_DIGITS)
                     weights_enabled = True
                     print(f"  Initialized class weights (past threshold): {[f'{w:.3f}' for w in class_weights if w > 0]}") # Only print non-zero weights
                 else:
                      print(f"  Dynamic weights will be enabled after generation {ENABLE_WEIGHTS_AFTER_GEN}.")
                 champion_generations_survived = 0 # Reset counter on load
            else:
                 print("  Warning: Champion file has incorrect shape/dtype. Ignoring.")
                 current_champion = None
        except Exception as e:
            print(f"  Warning: Could not load champion file ({e}). Ignoring.")
            current_champion = None

    # Define gen outside try block for finally clause
    gen = start_generation - 1

    try:
        for gen in range(start_generation, NUM_GENERATIONS):
            gen_start_time = time.time()
            print(f"\n--- Starting Generation {gen + 1}/{NUM_GENERATIONS} ---")

            # --- Check if we need to enable weights this generation ---
            if not weights_enabled and gen >= ENABLE_WEIGHTS_AFTER_GEN:
                 if current_champion is not None:
                     print(f"*** Enabling dynamic class weights starting this generation ({gen + 1}) ***")
                     # Calculate initial weights based on the current champion's validation performance
                     _, current_champ_per_class, _ = evaluate_champion_robust(current_champion)
                     class_weights = update_class_weights(current_champ_per_class, None, gen, TARGET_DIGITS)
                     weights_enabled = True
                 else:
                     # This case should ideally not happen if training starts from 0 or loads a champ
                     print(f"Warning: Trying to enable weights at gen {gen+1}, but no champion yet. Weights remain disabled.")

            # Prepare the weights to pass to C++ (10 elements, 0 for non-targets)
            cpp_class_weights = None # Default to None (uniform weights)
            if weights_enabled and class_weights is not None:
                cpp_class_weights = class_weights.tolist() # Convert full numpy array (with zeros) to list
                print(f"  Using dynamic class weights (non-zero): {[f'{digit}:{w:.3f}' for digit, w in enumerate(cpp_class_weights) if digit in TARGET_DIGITS]}")
            else:
                 print("  Using uniform class weights.")


            # 1. Evaluate fitness on GPU (Pass weights - potentially None or list of 10)
            fitness_scores = fast_network.evaluate_population_gpu(
                population,
                MINI_BATCH_SIZE,
                class_weights=cpp_class_weights # Pass current weights
            )
            fitness_scores = np.array(fitness_scores, dtype=np.float32)

            # 2. Find best and Evaluate Champion (on filtered validation set)
            best_fitness_this_gen = np.max(fitness_scores)
            if not np.isfinite(best_fitness_this_gen):
                print("Warning: Non-finite fitness scores encountered. Skipping generation.")
                continue

            best_individual_idx = np.argmax(fitness_scores)
            best_individual = population[best_individual_idx].copy()

            eval_start_time = time.time()
            candidate_accuracy, candidate_per_class, _ = evaluate_champion_robust(best_individual)
            eval_time = time.time() - eval_start_time

            # --- Champion Selection Logic & Update Weights ---
            new_champ_flag = False
            improvement = 0.0
            if current_champion is None:
                current_champion = best_individual
                current_champion_validation_accuracy = candidate_accuracy
                champion_generations_survived = 0
                new_champ_flag = True
                improvement = candidate_accuracy * 100
                # Update weights IF weights are enabled
                if weights_enabled:
                    class_weights = update_class_weights(candidate_per_class, class_weights, gen, TARGET_DIGITS)
            elif candidate_accuracy > current_champion_validation_accuracy + CHAMPION_REPLACEMENT_THRESHOLD:
                improvement = (candidate_accuracy - current_champion_validation_accuracy) * 100
                current_champion = best_individual
                current_champion_validation_accuracy = candidate_accuracy
                champion_generations_survived = 0
                new_champ_flag = True
                # Update weights IF weights are enabled
                if weights_enabled:
                    class_weights = update_class_weights(candidate_per_class, class_weights, gen, TARGET_DIGITS)
            else:
                champion_generations_survived += 1
                # Optional: Update weights even if champion defends, based on its own performance?
                # if weights_enabled:
                #    _, current_champ_per_class, _ = evaluate_champion_robust(current_champion) # Re-eval needed
                #    class_weights = update_class_weights(current_champ_per_class, class_weights, gen, TARGET_DIGITS)


            # --- (3. Breed New Generation unchanged) ---
            breed_start_time = time.time()
            champion_to_pass = current_champion if current_champion is not None else best_individual
            population = fast_network.breed_new_generation(
                population, fitness_scores, champion_to_pass,
                ELITISM_COUNT, TOURNAMENT_SIZE, MUTATION_RATE,
                NUM_TRANSFORMERS, BITS_PER_TRANSFORMER, CHROMOSOME_LENGTH
            )
            breed_time = time.time() - breed_start_time

            # --- (Reporting unchanged, but accuracy is on filtered set) ---
            gen_time = time.time() - gen_start_time
            print("\n" + "="*50)
            print(f"Generation {gen + 1} complete in {gen_time:.2f} seconds.")
            pop_eval_time = max(0, gen_time - eval_time - breed_time)
            print(f"  GPU Pop Eval: {pop_eval_time:.2f}s | Champ Eval: {eval_time:.2f}s | CPU Breed: {breed_time:.2f}s")
            print(f"  Best Evolution Fitness (Weighted): {best_fitness_this_gen*100:.2f}%") # Note: This fitness is weighted if weights are active
            print(f"  Candidate Validation Acc (Unweighted, Filtered Set): {candidate_accuracy*100:.2f}%")
            if new_champ_flag:
                 is_initial_champion = (gen == 0) or (gen == start_generation and start_generation > 0)
                 if not is_initial_champion:
                      print(f"  *** NEW CHAMPION (+{improvement:.2f}%) ***")
                 else:
                      print(f"  *** INITIAL CHAMPION ***")
            else:
                 champ_acc_str = f"{current_champion_validation_accuracy*100:.2f}%" if current_champion is not None else "N/A"
                 print(f"  Champion defended ({champ_acc_str}) (survived {champion_generations_survived} gen)")
            print("="*50)

            # --- (Backup Logic unchanged) ---
            if (gen + 1) % BACKUP_INTERVAL == 0:
                save_backup(population, gen + 1)
                if current_champion is not None:
                    np.save(CHAMPION_FILE, current_champion)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # --- (Final Saving unchanged) ---
        if current_champion is not None:
            print(f"\nSaving all-time best chromosome to '{CHAMPION_FILE}'...")
            np.save(CHAMPION_FILE, current_champion)
            # Report final accuracy on the filtered validation set
            final_acc, _, _ = evaluate_champion_robust(current_champion)
            print(f"Final champion accuracy (on filtered valid set): {final_acc*100:.2f}%")

        final_gen_num = gen + 1 if 'gen' in locals() and gen >= start_generation else start_generation
        if 'population' in locals() and population is not None:
            save_backup(population, final_gen_num)
        else:
            print("\nWarning: Population data not available for final backup.")

        print("\nTraining complete.")
