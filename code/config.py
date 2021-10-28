OUTPUT_SIZE = 388
PADDING_SIZE = 92

# 572 = 388 + 92 * 2
INPUT_SIZE = OUTPUT_SIZE + PADDING_SIZE*2

# Whether use lp_method as backend of U-Net
lp_method = False#True#True

_DATASET_NAMES = ['DRIVE', 'STARE', 'AV-WIDE', 'CHASEDB', 'VEVIO-FRAME', 'VEVIO-MOSAICS']

DATASET_NAME = _DATASET_NAMES[0]

epoch_round = 1000
#epoch_round = 100

validation_round = 5
#validation_round = 5

#dump_round = 2

need_dump = True