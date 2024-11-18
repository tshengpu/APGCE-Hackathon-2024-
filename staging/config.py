import loss

class DatasetConfig: 

    # Fault filtered Chendol 
    # RAW_SEISMIC_FOLDER = "../data/aug_raw_seismic_hasfault"
    # RAW_FAULT_FOLDER = "../data/aug_raw_fault_filter_hasfault"
    # RAW_HORIZON_FOLDER = "../data/aug_raw_horizon"
    # FAULT_MASK_FOLDER = "../data/aug_fault_mask_filter_hasfault"
    
    # Chendol 
    # RAW_SEISMIC_FOLDER = "../data/aug_raw_seismic"
    # RAW_FAULT_FOLDER = "../data/aug_raw_fault_filter"
    # RAW_HORIZON_FOLDER = "../data/aug_raw_horizon"
    # FAULT_MASK_FOLDER = "../data/aug_fault_mask_filter"
    
    # RAW_SEISMIC_FOLDER = "../data/aug_raw_seismic"
    # RAW_FAULT_FOLDER = "../data/aug_raw_fault"
    # RAW_HORIZON_FOLDER = "../data/aug_raw_horizon"
    # FAULT_MASK_FOLDER = "../data/aug_fault_mask"
    
    RAW_SEISMIC_FOLDER = "../data/raw_seismic"
    RAW_FAULT_FOLDER = "../data/raw_fault"
    RAW_HORIZON_FOLDER = "../data/raw_horizon"
    FAULT_MASK_FOLDER = "../data/fault_mask"
    DATA_FOLDER = "../data"

class DataProcessingConfig: 
    # IMAGE_SIZE = (224, 224)
    IMAGE_SIZE = (512, 512)
    # IMAGE_SIZE = (1024, 1024)
    PIXEL_CUTOFF_THRESHOLD = 0.5

class ModelParameters:
    # Feel free to tune these parameters for training
    EPOCHS = 3
    # EPOCHS = 5
    # EPOCHS = 20
    # EPOCHS = 50

    # BATCH_SIZE = 12 # (?)
    
    # BATCH_SIZE = 10 (memory break)
    # BATCH_SIZE = 9 # (best)
    # BATCH_SIZE = 8 # (better)
    # BATCH_SIZE = 6 # (Good)
    # BATCH_SIZE = 3
    BATCH_SIZE = 2
    
    # LEARNING_RATE = 0.00001  # (Too slow)
    LEARNING_RATE = 0.0001  # (Best)  
    # LEARNING_RATE = 0.001 # (Faster)
    PRETRAINED = True
    LOGGING = True # Record loss for each epoch
    
    # CRITERION = loss.DiceLoss() # Check loss.py for all available losses
    # CRITERION = loss.FocalLoss() # Check loss.py for all available losses
    CRITERION = loss.FocalLoss_Revised(alpha=0.75, gamma=2.0) # Check loss.py for all available losses
    # CRITERION = loss.CombinedDiceFocalLoss() # Check loss.py for all available losses
    
    EVAL_METRIC = 'batch_pix_accuracy' # batch_pix_accuracy or batch_intersection_union
    # EVAL_METRIC = 'batch_intersection_union'
    EVAL_METRIC = 'weighted_f1' # custom weighted f1 


class EvalConfig:
    F1_SCORE_CLASS_WEIGHT = [0.1, 0.9] # Background, Fault
    