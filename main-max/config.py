import loss

class DatasetConfig: 
    RAW_SEISMIC_FOLDER = "../data/aug_raw_seismic"
    RAW_FAULT_FOLDER = "../data/aug_raw_fault"
    RAW_HORIZON_FOLDER = "../data/aug_raw_horizon"
    FAULT_MASK_FOLDER = "../data/aug_fault_mask"
    # RAW_SEISMIC_FOLDER = "../data/raw_seismic"
    # RAW_FAULT_FOLDER = "../data/raw_fault"
    # RAW_HORIZON_FOLDER = "../data/raw_horizon"
    # FAULT_MASK_FOLDER = "../data/fault_mask"
    DATA_FOLDER = "../data"

class DataProcessingConfig: 
    # IMAGE_SIZE = (224, 224)
    IMAGE_SIZE = (512, 512)
    # IMAGE_SIZE = (1024, 1024)
    PIXEL_CUTOFF_THRESHOLD = 0.5

class ModelParameters:
    # Feel free to tune these parameters for training
    EPOCHS = 5

    # BATCH_SIZE = 10 (memory break)
    BATCH_SIZE = 9 # (best)
    # BATCH_SIZE = 8 # (better)
    # BATCH_SIZE = 6 # (Good)
    # BATCH_SIZE = 3
    # BATCH_SIZE = 2
    LEARNING_RATE = 0.00001  # (Good)
    # LEARNING_RATE = 0.001
    PRETRAINED = True
    LOGGING = True # Record loss for each epoch
    CRITERION = loss.FocalLoss() # Check loss.py for all available losses
    EVAL_METRIC = 'batch_pix_accuracy' # batch_pix_accuracy or batch_intersection_union
    # EVAL_METRIC = 'batch_intersection_union'
    