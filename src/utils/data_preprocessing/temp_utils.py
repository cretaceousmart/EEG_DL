import mne
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
from io import BytesIO
from tqdm import tqdm
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import gc # garbage collection
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Scaling factor is not defined.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Physical range is not defined.*')

from memory_profiler import profile
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

def read_edf(file_path):
    """
    read edf file to obtain raw signal
    """
    raw = mne.io.read_raw_edf(file_path, preload=True, encoding='latin1', verbose='WARNING')
    return raw

def rename_edf(raw):
    """
    reanme the channel name of raw signal
    """
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
    
    # pick the target channel
    target_EEG_channel = list(channel_mapping.keys())
    raw.pick(target_EEG_channel)
    
    # rename the channel name
    raw.rename_channels(channel_mapping)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='warn')
    return raw

def obtain_processed_raw(file_path):
    raw = read_edf(file_path)
    raw = rename_edf(raw)
    raw.filter(1, 60, fir_design='firwin',verbose='WARNING')
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

# @profile
def obtain_single_topomap(raw, picks, info, start_time, end_time, vmax, is_energy=True):
    """
    Obtain the topomap for a given time period and return the figure object for later use.
    """
    data, times = raw.get_data(picks=picks, start=start_time, stop=end_time, return_times=True)
    # if is_energy: 
    data = np.sum(data**2, axis=1)
    #     # data = np.sum(abs(data), axis=1)
    # else:
    #     data = data[:, 0]

    fig, ax = plt.subplots()
    mne.viz.plot_topomap(data, info, axes=ax, show=False,ch_type='eeg',outlines='head',vlim=(0,vmax))  # 设置show=False以获取图形对象
    plt.close(fig)  # 关闭图形以避免显示
    return fig  # 返回Figure对象以供后续查看



def obtain_multi_topomap(raw, picks, info, output_file_name, eeg_file_name, fig_size=128, vmax = 0.1, is_energy = True, is_test = None):

    current_path = os.path.abspath('')
    root_dir = os.path.dirname(os.path.dirname(current_path))
    output_dir = os.path.join(root_dir, 'EEG_DL_CLASSIFIER','src','data', output_file_name, eeg_file_name)

    print(f"Jie Log: output_dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    sample_rate = int(raw.info['sfreq'])

    index = 3600 if not is_test else 10
    
    for i in tqdm(range(0,index), desc=f"Obtain topomap for {eeg_file_name}"):
        start_time = int(i * sample_rate)  # 将秒转换为样本数
        end_time = int((i + 1) * sample_rate)  # 将秒转换为样本数

        # print(f"Jie Log: start_time: {start_time}, end_time: {end_time}, sample_rate: {sample_rate}, int(start_time/sample_rate): {int(start_time/sample_rate)}")
        fig = obtain_single_topomap(raw=raw, picks=picks, info=info, vmax=vmax, start_time=start_time, end_time=end_time, is_energy=is_energy)

        output_path = os.path.join(output_dir, f"{eeg_file_name}_{int(start_time/sample_rate)}.png")
        temp_path = f'../src/data/{output_file_name}/temp.png'     

        fig.savefig(temp_path, dpi=100)  # 保存原始图形
        with Image.open(temp_path) as _img:
            _img_resized = _img.resize((fig_size, fig_size), Image.Resampling.LANCZOS)
            _img_resized.save(output_path)
        
        plt.close(fig)
        del fig
        # 每1000次进行一次垃圾回收：
        if i % 500 == 0:
            gc.collect()
        os.remove(temp_path)

# @profile
def obtain_multi_topomap_new(raw, picks, info, output_file_name, eeg_file_name, fig_size=128, vmax=0.1, is_energy=True, is_test=None):
    current_path = os.path.abspath('')
    root_dir = os.path.dirname(os.path.dirname(current_path))
    output_dir = os.path.join(root_dir, 'EEG_DL_CLASSIFIER', 'src', 'data', output_file_name, eeg_file_name)

    print(f"Jie Log: output_dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    sample_rate = int(raw.info['sfreq'])
    index = 3600 if not is_test else 10

    for i in tqdm(range(0,index), desc=f"Obtain topomap for {eeg_file_name}"):
        start_time = i * sample_rate
        end_time = (i + 1) * sample_rate
        
        fig = obtain_single_topomap(raw=raw, picks=picks, info=info, start_time=start_time, end_time=end_time, vmax=vmax, is_energy=is_energy)
        
        output_path = os.path.join(output_dir, f"{eeg_file_name}_{i}.png")
        
        # # 直接保存调整大小后的图像，避免使用临时文件
        # fig.canvas.draw()  # 渲染图形
        # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) # 将画布转换为numpy数组
        # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,)) # 重新排列数组形状
        # img_pil = Image.fromarray(img) # 从numpy数组创建PIL图像
        # img_resized = img_pil.resize((fig_size, fig_size), Image.Resampling.LANCZOS)
        # img_resized.save(output_path)
        # plt.close(fig)
        # del fig
        # if i % 500 == 0:
        #     gc.collect()

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=30)
        buf.seek(0)
        img = Image.open(buf)
        img_resized = img.resize((fig_size, fig_size), Image.Resampling.LANCZOS)
        output_path = os.path.join(output_dir, f"{eeg_file_name}_{i}.png")
        img_resized.save(output_path)
        
        # 关闭图形对象和清理
        plt.close(fig)
        buf.close()
        del fig, img, img_resized, buf
        if i % 360 == 0:
            gc.collect()

# ------------------- 以下是数据可视化部分 ------------------- #

def visulize_EEG_power(file_path,plotly=False):
    raw = obtain_processed_raw(file_path)
    picks = obtain_picks(raw)
    info = obtain_eeg_info(raw, picks)

    time_length = 3600 # 3600 seconds
    sample_rate = raw.info['sfreq']
    energy_list = np.zeros((time_length, 16))
    for i in range(0,time_length): #先测试0到5秒
        start_time = int(i * sample_rate)  # 将秒转换为样本数
        end_time = int((i + 1) * sample_rate)  # 将秒转换为样本数

        data, times = raw.get_data(picks=picks, start=start_time, stop=end_time, return_times=True)
        energy = np.sum(data**2, axis=1)
        energy_list[i] = energy

    print(f"这是file_path: {file_path}的EEG能量时间趋势图")
    if not plotly:
        fig, ax = plt.subplots()
        ax.plot(energy_list)
        ax.set(xlabel='time (s)', ylabel='energy', title='第{file_path}个患者 energy of 16 channels')
        
        # 设置y轴的范围为0到0.0001，且按照0.00001的间隔显示
        ax.set_ylim(0, 0.0001)
        # 把每个通道的名字放在图的右边
        ax.legend(picks, loc='center left', bbox_to_anchor=(1, 0.5))
        # 设置图像的宽高为：
        fig.set_size_inches(12, 10)
        ax.grid()
        plt.show()
        return energy_list
    else:
        # 使用plotly绘制可交互的折线图并在HTLM中显示
        pio.renderers.default = "browser"
        fig = go.Figure()
        for i in range(16):
            fig.add_trace(go.Scatter(x=np.arange(0, time_length), y=energy_list[:, i], mode='lines', name=raw.info['ch_names'][i]))

        # 设置y轴的范围为0到0.0001，且按照0.00001的间隔显示

        fig.update_layout(title=f'Patient No.{file_path}: energy of 16 channels', xaxis_title='time (s)', 
                          yaxis_title='energy', xaxis=dict(tickmode='linear', tick0=0, dtick=300), yaxis=dict(range=[0, 0.0001]))
        
        
        # 在浏览器中显示图像
        fig.show()
        # return energy_list


def compare_seizure_non_seizure(file_index, y_max):
    edf_file_path = rf'../src/data/EEG_DATASET/{file_index}/{file_index}.edf'
    label_file_path = rf'../src/data/EEG_DATASET/{file_index}/{file_index}.xlsx'

    with pd.ExcelFile(label_file_path) as xls:
        label_df = pd.read_excel(xls, 'Sheet1', header=None)
        label_df.rename(columns={0: 'label'}, inplace=True)

    # 计算label为0的行的个数，然后选取所有的label为0的行
    label_0_count = label_df[label_df['label'] == 0].shape[0]

    # 分别选取所有为0的行 以及 label_0_count个为1的行 分别储存他们的行号
    label_0_index = label_df[label_df['label'] == 0].index
    label_1_index = label_df[label_df['label'] == 1].index[:label_0_count]

    # 从eeg信号中选取对于时间点的数据

    raw = obtain_processed_raw(edf_file_path)
    picks = obtain_picks(raw)
    info = obtain_eeg_info(raw, picks)

    time_length = 3600 # 3600 seconds
    sample_rate = raw.info['sfreq']

    energy_list = np.zeros((2,label_0_count, 16))


    for i, index in enumerate(label_0_index):
        start_time, end_time = int(index * sample_rate), int((index + 1) * sample_rate)
        data, times = raw.get_data(picks=picks, start=start_time, stop=end_time, return_times=True)
        energy = np.sum(data**2, axis=1)
        energy_list[0][i] = energy

    for i, index in enumerate(label_1_index):
        start_time, end_time = int(index * sample_rate), int((index + 1) * sample_rate)
        data, times = raw.get_data(picks=picks, start=start_time, stop=end_time, return_times=True)
        energy = np.sum(data**2, axis=1)
        energy_list[1][i] = energy
    

    # 创建一个2行1列的子图布局
    fig = make_subplots(rows=1, cols=2)

    # 第一个子图
    for i in range(16):
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, energy_list.shape[1]),  # 时间点数可能需要根据您的数据进行调整
                y=energy_list[0, :, i],
                mode='lines',
                name=raw.info['ch_names'][i]
            ),
            row=1, col=1
        )

    # 第二个子图
    for i in range(16):
        fig.add_trace(
            go.Scatter(
                x=np.arange(0, energy_list.shape[1]),  # 时间点数可能需要根据您的数据进行调整
                y=energy_list[1, :, i],
                mode='lines',
                name=raw.info['ch_names'][i]
            ),
            row=1, col=2
        )

    # 设置y轴的范围为0到0.0001，且按照0.00001的间隔显示，并更新布局
    fig.update_layout(
        title=f'Patient No.{edf_file_path}: energy of 16 channels', 
        xaxis_title='time (s)', 
        yaxis_title='energy', 
        xaxis=dict(tickmode='linear', tick0=0, dtick=300),
        xaxis2=dict(tickmode='linear', tick0=0, dtick=300)
    )

    # 更新每个子图的Y轴设置
    fig.update_yaxes(range=[0, y_max], dtick=y_max/10, row=1, col=1)
    fig.update_yaxes(range=[0, y_max], dtick=y_max/10, row=1, col=2)

    # 在浏览器中显示图像
    fig.show()




# ------------------- 以下legacy code------------------- #

def prepare_data(eeg_file_name, handle_data_imbalance = False, is_test = True):
    # read label for each image:
    with pd.ExcelFile(rf'../src/data/EEG_DATASET/{eeg_file_name}/{eeg_file_name}.xlsx') as xls:
        label_df = pd.read_excel(xls, 'Sheet1',header=None)
        label_df.rename(columns={0: 'label'}, inplace=True)
    
    if is_test: label_df = label_df[:100]

    # read image data: 
    images = []
    image_folder = rf'../src/data/output_image/{eeg_file_name}/'

    # 加载并处理每张图片
    label_size = 3600 if not is_test else 100 

    for image_id in range(label_size):
        image_path = os.path.join(image_folder, f'{eeg_file_name}_{image_id}.png')
        with Image.open(image_path) as img:
            img_gray = img.convert('L')  # 转换为灰度图像
            img_array = np.array(img_gray).flatten()  # 展平为一维向量
            images.append(img_array)

    # 将图片数据转换为 numpy 数组
    images_np = np.array(images)
    # 获取标签
    labels = label_df['label'].to_numpy()

    # 划分训练集和测试集
    print(f"Jie log: images_np.shape: {images_np.shape}, labels.shape: {labels.shape}")
    if handle_data_imbalance:
        X_train, X_test, y_train, y_test = train_test_split(images_np, labels, test_size=0.3, random_state=42, stratify=labels)
    else:
        X_train, X_test, y_train, y_test = train_test_split(images_np, labels, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test
    


def prepare_data_from_multi_file(eeg_file_names, handle_data_imbalance = False, is_test = True):
    all_images = []
    all_labels = []

    for eeg_file_name in eeg_file_names:
        # 读取每个图像的标签
        with pd.ExcelFile(rf'../src/data/EEG_DATASET/{eeg_file_name}/{eeg_file_name}.xlsx') as xls:
            label_df = pd.read_excel(xls, 'Sheet1', header=None)
            label_df.rename(columns={0: 'label'}, inplace=True)

        if is_test:
            label_df = label_df[:100]

        # 读取图像数据
        image_folder = rf'../src/data/output_image/{eeg_file_name}/'
        label_size = 3600 if not is_test else 100 

        for image_id in range(label_size):
            image_path = os.path.join(image_folder, f'{eeg_file_name}_{image_id}.png')
            with Image.open(image_path) as img:
                img_gray = img.convert('L')  # 转换为灰度图像
                img_array = np.array(img_gray).flatten()  # 展平为一维向量
                all_images.append(img_array)

        # 将标签追加到总列表
        all_labels.extend(label_df['label'].tolist())

    # 将图像数据转换为 numpy 数组
    all_images_np = np.array(all_images)
    # 将标签转换为 numpy 数组
    all_labels_np = np.array(all_labels)

    # 划分训练集和测试集
    print(f"Jie log: all_images_np.shape: {all_images_np.shape}, all_labels_np.shape: {all_labels_np.shape}")
    if handle_data_imbalance:
        X_train, X_test, y_train, y_test = train_test_split(all_images_np, all_labels_np, test_size=0.3, random_state=42, stratify=all_labels_np)
    else:
        X_train, X_test, y_train, y_test = train_test_split(all_images_np, all_labels_np, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


    
    
        