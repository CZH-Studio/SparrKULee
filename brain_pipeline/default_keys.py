from enum import StrEnum

from typing import List, Optional


class DefaultKeys(StrEnum):
    #########
    # 根目录 #
    #########
    BASE_DIR = "input_base_dir"
    """数据集的根目录"""

    ##############
    # 刺激文件相关 #
    ##############
    INPUT_STIMULI_PATH = "input_stimuli_path"
    """刺激文件的路径"""
    INPUT_STIMULI_DATA = "input_stimuli_data"
    """刺激文件数据"""
    INPUT_STIMULI_SR = "input_stimuli_sr"
    """刺激文件采样率"""
    INPUT_STIMULI_TRIGGER_PATH = "input_stimuli_trigger_path"
    """刺激文件触发信号的路径"""
    INPUT_STIMULI_TRIGGER_DATA = "input_stimuli_trigger_data"
    """刺激文件触发信号数据"""
    INPUT_STIMULI_TRIGGER_SR = "input_stimuli_trigger_sr"
    """刺激文件触发信号采样率"""
    INPUT_STIMULI_NOISE_PATH = "input_stimuli_noise_path"
    """刺激文件噪声的路径"""
    INPUT_STIMULI_NOISE_DATA = "input_stimuli_noise_data"
    """刺激文件噪声数据"""
    INPUT_STIMULI_NOISE_SR = "input_stimuli_noise_sr"
    """刺激文件噪声采样率"""

    ##########
    # EEG相关 #
    ##########
    INPUT_EEG_PATH = "input_eeg_path"
    """EEG文件的路径"""
    INPUT_EEG_DATA = "input_eeg_data"
    """EEG数据"""
    INPUT_EEG_SR = "input_eeg_sr"
    """EEG采样率"""
    INPUT_EEG_TRIGGER = "input_eeg_trigger"
    """EEG触发信号"""

    #############
    # APR文件相关 #
    #############
    INPUT_APR_PATH = "input_apr_path"
    """APR文件的路径"""
    INPUT_APR_DATA = "input_apr_data"
    """APR数据"""
    INPUT_APR_DATA_SNR = "input_apr_data_snr"
    """APR数据与噪声的信噪比"""

    #############
    # TSV格式相关 #
    #############
    INPUT_TSV_STIMULATION_PATH = "input_tsv_stimulation_path"
    """TSV格式的保存刺激的路径"""
    INPUT_TSV_STIMULATION_DATA = "input_tsv_stimulation_data"
    """TSV格式的刺激数据"""
    INPUT_TSV_EVENTS_PATH = "input_tsv_events_path"
    """TSV格式的保存事件的路径"""
    INPUT_TSV_EVENTS_DATA = "input_tsv_events_data"
    """TSV格式的事件数据"""

    ##############
    # 中间结果相关 #
    ##############
    TMP_EEG_DATA = "tmp_data"
    """中间产物临时EEG数据"""
    TMP_STIMULI_DATA = "tmp_stimuli_data"
    """中间产物临时刺激数据"""
    FILTERED_DATA = "filtered_data"
    """滤波后的数据"""
    ENVELOPE_DATA = "envelope_data"
    """语音包络"""
    RESAMPLED_DATA = "resampled_data"
    """重采样后的数据"""
    OUTPUT_STATUS = "output_status"
    """在context中，输出结果的状态（成功与否）"""

    ##########
    # 垃圾回收 #
    ##########
    GC_COLLECT = "gc_collect"

KeyTypeNotNone = List[str]
KeyTypeNoneOk = Optional[List[str]]
