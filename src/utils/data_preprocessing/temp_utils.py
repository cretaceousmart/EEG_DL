import mne
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os 
from tqdm import tqdm

def read_edf(file_path):
    """
    read edf file to obtain raw signal
    """
    raw = mne.io.read_raw_edf(file_path, preload=True, encoding='latin1')
    return raw

def rename_edf(raw):
    """
    reanme the channel name of raw signal
    """
    EEG_channels_name = [chanel for chanel in raw.ch_names if "EEG" in chanel]
    assert len(EEG_channels_name) == 16, f"EEG channels number is not 16 but {len(EEG_channels_name)}"
    raw.pick(EEG_channels_name)
    # rename the channel name
    channel_mapping = {
        'EEG C3-Ref': 'C3',
        'EEG C4-Ref': 'C4',
        'EEG O1-Ref': 'O1',
        'EEG O2-Ref': 'O2',
        'EEG Fp1-Ref': 'Fp1',
        'EEG Fp2-Ref': 'Fp2',
        'EEG T3-Ref': 'T7',  # T3 在标准10-20系统中通常对应于T7
        'EEG T4-Ref': 'T8',  # T4 在标准10-20系统中通常对应于T8
        'EEG F3-Ref': 'F3',
        'EEG F4-Ref': 'F4',
        'EEG P3-Ref': 'P3',
        'EEG P4-Ref': 'P4',
        'EEG F7-Ref': 'F7',
        'EEG F8-Ref': 'F8',
        'EEG T5-Ref': 'P7',  # T5 在标准10-20系统中通常对应于P7
        'EEG T6-Ref': 'P8'   # T6 在标准10-20系统中通常对应于P8
    }

    raw.rename_channels(channel_mapping)
    
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='warn')
    return raw

def obtain_picks(raw):
    EEG_channels_name = [chanel for chanel in raw.ch_names if "EEG" in chanel]
    picks = mne.pick_channels(raw.info['ch_names'], include=EEG_channels_name)
    return picks


def obtain_eeg_info(raw, picks):
    """
    obtain eeg info
    """
    return mne.pick_info(raw.info, sel=picks)



# def obtain_raw_topomap(raw, picks, info, start_time, end_time):
#     """
#     Obtain the topomap for a given time period and return the figure object for later use.
#     """
#     raw.filter(1, 60, fir_design='firwin') # 滤波: 1-60Hz
#     data, times = raw.get_data(picks=picks, start=start_time, stop=end_time, return_times=True)
#     fig, ax = plt.subplots()
#     mne.viz.plot_topomap(data[:, 0], info, axes=ax, show=False)  # 设置show=False以获取图形对象
#     plt.close(fig)  # 关闭图形以避免显示
#     return fig  # 返回Figure对象以供后续查看


# def obtain_energy_topomap(raw, picks, info, start_time, end_time):
#     """
#     Obtain the topomap for a given time period and return the figure object for later use.
#     """
#     raw.filter(1, 60, fir_design='firwin') # 滤波: 1-60Hz
#     data, times = raw.get_data(picks=picks, start=start_time, stop=end_time, return_times=True)
#     energy = np.sum(data**2, axis=1)

#     fig, ax = plt.subplots()
#     mne.viz.plot_topomap(energy, info, axes=ax, show=False)  # 设置show=False以获取图形对象
#     plt.close(fig)  # 关闭图形以避免显示
#     return fig  # 返回Figure对象以供后续查看


def obtain_single_topomap(raw, picks, info, start_time, end_time, is_energy=True):
    """
    Obtain the topomap for a given time period and return the figure object for later use.
    """
    data, times = raw.get_data(picks=picks, start=start_time, stop=end_time, return_times=True)
    if is_energy: 
        data = np.sum(data**2, axis=1)
    else:
        data = data[:, 0]

    fig, ax = plt.subplots()
    mne.viz.plot_topomap(data, info, axes=ax, show=False)  # 设置show=False以获取图形对象
    plt.close(fig)  # 关闭图形以避免显示
    return fig  # 返回Figure对象以供后续查看



def obtain_multi_topomap(raw, picks, info, eeg_file_name, fig_size=128, is_energy = True, is_test = None):
    output_dir = rf'D:\Research\EEG\EEG_DL_Classifier\tests\{eeg_file_name}'
    os.makedirs(output_dir, exist_ok=True)
    sample_rate = raw.info['sfreq']

    index = 3600 if not is_test else 10
    for i in tqdm(range(index), desc=f"Obtain topomap for {eeg_file_name}"):
        start_time = int(i * sample_rate)  # 将秒转换为样本数
        end_time = int((i + 1) * sample_rate)  # 将秒转换为样本数
        fig = obtain_single_topomap(raw, picks, info, start_time=start_time, end_time=end_time, is_energy=is_energy)

        output_path = rf'D:\Research\EEG\EEG_DL_Classifier\tests\{eeg_file_name}\{eeg_file_name}_{int(start_time/sample_rate)}.png'
        temp_path = r'D:\Research\EEG\EEG_DL_Classifier\tests\temp.png'
        
        fig.savefig(temp_path, dpi=100)  # 保存原始图形
        with Image.open(temp_path) as _img:
            _img_resized = _img.resize((fig_size, fig_size), Image.Resampling.LANCZOS)
            _img_resized.save(output_path)
        
        plt.close(fig)
        os.remove(temp_path)




    


    
    
        