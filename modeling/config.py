import loss

class DatasetConfig: 
    RAW_SEISMIC_FOLDER = "../data/aug_raw_seismic_hasfault"
    RAW_FAULT_FOLDER = "../data/aug_raw_fault_filter_hasfault"
    FAULT_MASK_FOLDER = "../data/aug_fault_mask_filter_hasfault"
    DATA_FOLDER = "../data"

class DataProcessingConfig: 
    IMAGE_SIZE = (512, 512)
    PIXEL_CUTOFF_THRESHOLD = 0.5

class ModelParameters:
    # Feel free to tune these parameters for training
    EPOCHS = 3

    BATCH_SIZE = 2
    LEARNING_RATE = 0.0001
    PRETRAINED = True
    LOGGING = True # Record loss for each epoch
    CRITERION = loss.FocalLoss_Revised(alpha=0.75, gamma=2.0) # Check loss.py for all available losses
    EVAL_METRIC = 'weighted_f1' # batch_pix_accuracy or batch_intersection_union

class EvalConfig:
    F1_SCORE_CLASS_WEIGHT = [0.1, 0.9] # Background, Fault
    