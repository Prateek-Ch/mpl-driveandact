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
    
    data_batch = [item[0] for item in batch]  # Extract data (video clips)
    label_batch = [item[1] for item in batch]  # Extract labels
    
    # Convert label_batch elements from numpy.ndarray to torch.Tensor
    label_batch = [torch.tensor(label) if isinstance(label, np.ndarray) else label for label in label_batch]
    
    # Find the max dimensions across all clips
    max_frames = max([x.shape[2] for x in data_batch])  # Max number of frames
    max_height = max([x.shape[3] for x in data_batch])  # Max height
    max_width = max([x.shape[4] for x in data_batch])   # Max width

    # Padding logic for frames, height, and width
    padded_data_batch = []
    for clip in data_batch:
        batch_size, channels, frames, height, width = clip.shape
        
        # Padding for frames, height, and width
        if frames < max_frames:
            pad_frames = max_frames - frames
            clip = torch.nn.functional.pad(clip, (0, 0, 0, 0, 0, pad_frames), "constant", 0)

        if height < max_height or width < max_width:
            pad_height = max_height - height
            pad_width = max_width - width
            clip = torch.nn.functional.pad(clip, (0, pad_width, 0, pad_height), "constant", 0)
        
        padded_data_batch.append(clip)
    
    # Stack the data batch into a single tensor
    data_tensor = torch.stack(padded_data_batch, 0)  # Stack the data into a single tensor
    
    # Pad labels to the same length (to the length of the longest label)
    max_label_length = max([label.size(0) for label in label_batch])  # Get the longest label length
    padded_label_batch = []
    for label in label_batch:
        padding_length = max_label_length - label.size(0)
        if padding_length > 0:
            padded_label = torch.cat([label, torch.full((padding_length,), -1)])  # Pad with -1 (or another value)
        else:
            padded_label = label
        padded_label_batch.append(padded_label)
    
    # Stack the padded labels
    label_tensor = torch.stack(padded_label_batch, 0)
    
    return data_tensor, label_tensor  # Return a tuple of data and labels
