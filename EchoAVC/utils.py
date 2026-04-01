import numpy as np

class Task():
    """Echocardiography interpretation task object."""
    def __init__(self, task_name, task_type, class_names, mean=np.nan):
        self.task_name = task_name
        self.task_type = task_type
        self.class_names = class_names  # ndarray
        self.class_indices = np.arange(class_names.size)
        self.mean = mean

def merge_task_dicts(d):
    merged_dict = {}

    # Iterate through each dictionary in the list
    for dictionary in d:
        # Iterate through keys in the dictionary
        for key, value in dictionary.items():
            if key in merged_dict:
                # Merge lists if key is present in both merged_dict and the current dictionary
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key in merged_dict[key]:
                            merged_dict[key][sub_key] += sub_value
                        else:
                            merged_dict[key][sub_key] = sub_value
                else:
                    merged_dict[key] += value
            else:
                # Otherwise, assign the corresponding dictionary
                merged_dict[key] = value

    return merged_dict

