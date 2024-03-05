import mne
import numpy as np
print(mne.__version__)
print(np.__version__)
import sys
import src.utils.data_preprocessing.temp_utils as utils


import gc


print("done")
import time 
output_file_name = 'output_image_new'

start_time = time.time()
for file_index in range(10,11):
        file_path = rf'D:\Research\EEG\EEG_DL_Classifier\src\data\EEG_DATASET\{file_index}\{file_index}.edf'
        raw = utils.obtain_processed_raw(file_path)

        picks = utils.obtain_picks(raw)
        info = utils.obtain_eeg_info(raw, picks)
        utils.obtain_multi_topomap_new(raw, picks, info, output_file_name=output_file_name, eeg_file_name=str(file_index), fig_size=128, vmax=0.00003, is_energy=True, is_test=False)
        del raw, picks, info
        gc.collect()

end_time = time.time()
print(f"Time cost: {end_time - start_time} seconds")