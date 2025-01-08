import torch
import numpy as np
from framework_activity_recognition.processing import normalize_color_input_zero_center_unit_range, \
    normalize_color_input_zero_center_unit_range_per_channel,unit_range_zero_center_to_unit_range_zero_min,\
        center_crop, random_crop, random_select, random_horizontal_flip

class normalizeColorInputZeroCenterUnitRange(object):
    def __init__(self, max_val = 255.0):

        self.max_val = max_val


    def __call__(self, input_tensor):
        result = normalize_color_input_zero_center_unit_range(input_tensor, max_val = self.max_val)

        return result

class normalizeColorInputZeroCenterUnitRangeChannelWise(object):

    def __call__(self, input_tensor):
        result = normalize_color_input_zero_center_unit_range_per_channel(input_tensor)

        return result

class unitRangeZeroCenterToUnitRangeZeroMin(object):

    def __call__(self, input_tensor):
        result = unit_range_zero_center_to_unit_range_zero_min(input_tensor)

        return result

class CenterCrop(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, input_tensor):

        result = center_crop(input_tensor, self.height, self.width)

        return result


class RandomCrop(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, input_tensor):
        # currently results in a shape of (16, 16, 224, 224, 3)
        result = random_crop(input_tensor, self.height, self.width)

        return result


class RandomSelect(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, input_tensor):

        result = random_select(input_tensor, self.n)

        return result


class RandomHorizontalFlip(object):
    def __init__(self):
        super().__init__()

    def __call__(self, input_tensor):

        result = random_horizontal_flip(input_tensor)

        return result


class ToTensor(object):
    """"
        Takes the input tensor which in case of 5D is of the shape (16, 16, 224, 224, 3)
        Outputs a  resultant tensor of the shape (3, 16, 16, 224, 224)
    """
    def __call__(self, input_tensor):
        if input_tensor.ndim == 4:
            # Input tensor is in the shape (num_frames, height, width, channels)
            result = input_tensor.transpose(3, 0, 1, 2)  # (channels, num_frames, height, width)
        elif input_tensor.ndim == 5:
            # Input tensor is in the shape (num_windows, num_frames, height, width, channels)
            # current input tensor shape is (16, 16, 224, 224, 3)
            result = input_tensor.transpose(4, 0, 1, 2, 3)  # (channels, num_windows, num_frames, height, width)
        else:
            raise ValueError(f"Unsupported tensor shape: {input_tensor.shape}")

        result = np.float32(result)

        return torch.from_numpy(result)

def custom_collate_fn(batch):
    """Custom collate function to handle padding of variable-sized video clips and labels."""
    data_batch = [item[0] for item in batch]  # Extract windows
    label_batch = [item[1] for item in batch]  # Extract frame-wise labels

    # Determine max dimensions for padding
    max_frames = max([x.shape[0] for x in data_batch])  # Max frames in window
    max_height = max([x.shape[1] for x in data_batch])  # Height
    max_width = max([x.shape[2] for x in data_batch])   # Width

    # Pad data
    padded_data = []
    for frames in data_batch:
        pad_frames = max_frames - frames.shape[0]
        pad_height = max_height - frames.shape[1]
        pad_width = max_width - frames.shape[2]
        frames = np.pad(frames, ((0, pad_frames), (0, pad_height), (0, pad_width), (0, 0)), mode="constant")
        padded_data.append(torch.tensor(frames).permute(3, 0, 1, 2))  # (C, T, H, W)

    # Pad labels
    padded_labels = []
    for labels in label_batch:
        pad_len = max_frames - len(labels)
        padded_labels.append(torch.cat([torch.tensor(labels), torch.full((pad_len,), -1)]))  # -1 for padding

    return torch.stack(padded_data), torch.stack(padded_labels)
