import random


def shuffle_and_split_data(data, ratio, seed=0):
    """
    Shuffle the data list with a given seed and split it according to the given ratio.

    Parameters:
    - data: original list of data.
    - ratio: The ratio of the data to select after shuffling (0 < ratio <= 1).
    - seed: The random seed for reproducibility.

    Returns:
    - A tuple of two lists: the selected subset of data and the remaining data.
    """
    
    random.seed(seed)
    data_copy = data[:]
    
    random.shuffle(data_copy)
    
    split_index = int(len(data_copy) * (1-ratio))
    
    selected_data = data_copy[:split_index]
    remaining_data = data_copy[split_index:]
    
    return selected_data, remaining_data


def shuffle_and_split_data_in_group(data, ratio, seed=0):
    """
    Shuffle the data list with a given seed and split it according to the given ratio,
    ensuring that pMHC sequences in the training and validation sets do not overlap.

    Parameters:
    - data: original list of data, where each item is [dec_input_ids, output_labels, mask_for_loss, enc_input_ids, masked_pos].
    - ratio: The ratio of the data to select for validation (0 < ratio < 1).
    - seed: The random seed for reproducibility.

    Returns:
    - A tuple of two lists: the training data and the validation data.
    """
    random.seed(seed)

    # Group data by pMHC sequence
    pmhc_to_data = {}
    for item in data:
        enc_input_ids = item[3]  # Extract pMHC sequence (enc_input_ids)
        pmhc_seq = tuple(enc_input_ids)  # Use tuple as key for grouping
        if pmhc_seq not in pmhc_to_data:
            pmhc_to_data[pmhc_seq] = []
        pmhc_to_data[pmhc_seq].append(item)

    # Shuffle pMHC sequences
    pmhc_list = list(pmhc_to_data.keys())
    random.shuffle(pmhc_list)

    # Calculate the total number of data points
    total_data_points = len(data)
    valid_data_points = int(total_data_points * ratio)  # Number of data points for validation

    # Assign pMHC sequences to training or validation set
    train_batch = []
    valid_batch = []
    current_valid_points = 0

    for pmhc_seq in pmhc_list:
        data_list = pmhc_to_data[pmhc_seq]
        if current_valid_points + len(data_list) <= valid_data_points:
            # Assign this pMHC's data to validation set
            valid_batch.extend(data_list)
            current_valid_points += len(data_list)
        else:
            # Assign this pMHC's data to training set
            train_batch.extend(data_list)

    return train_batch, valid_batch