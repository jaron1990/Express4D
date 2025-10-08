import json
import os

FAMOS_EXPRESSION_LIST = ["anger", "bareteeth", "blow_cheeks", "cheeks_in",
                        "disgust", "eyebrow", "fear", "happiness",
                        "head_rotation_left_right", "head_rotation_up_down", "high_smile",
                        "jaw", "kissing", "lip_corners_down", "lips_back", "lips_up",
                        "mouth_down", "mouth_extreme", "mouth_middle", "mouth_open",
                        "mouth_side", "mouth_up", "rolling_lips",
                        "sadness", "sentence", "smile_closed",
                        "surprise", "wrinkle_nose"]

FLORENCE_EXPRESSION_LIST = ["Afraid", "Agony", "Amused", "Angry1", "Angry2", "Arrogant", "Ashamed", 
                            "Awe", "Bereft", "Bored", "Cheeky", "Concentrate", "Confident", 
                            "Confused", "Contempt", "Cool", "Desire", "Disgust", "Displeased", 
                            "Ditzy", "Dreamy", "Drunk1", "Drunk2", "Excitement", "Fear", 
                            "Fierce", "Flirting", "Frown", "Glare", "Happy", "Hot", "Hurt", 
                            "Ignore", "Ill", "Incredulous", "Innocent", "Irritated1", 
                            "Irritated2", "Kissy", "Laughing", "Moody", "Mourning", "Pain", 
                            "Pleased", "Pouting", "Pouty", "Rage", "Sad1", "Sad2", "Sarcastic", 
                            "Scream", "Serious", "Shock", "Silly", "Smile1", "Smile2", "Smile3", 
                            "Smile4", "Snarl", "Surprised", "Suspicious", "Terrified", "Tired1", 
                            "Tired2", "Triumph", "Unimpressed", "Upset", "Wink", "Worried", "Zen"]

COMA_EXPRESSION_LIST = ["bareteeth", "cheeks_in", "eyebrow", "high_smile",
                        "lips_back", "lips_up", "mouth_down", "mouth_extreme",
                        "mouth_middle", "mouth_open", "mouth_side",
                        "mouth_up"]


def generate_json_of_data(root_dir, dataset_name, output_folder):
    """
    This function will generate a json file of the data in the root_dir
    :param root_dir:
    :param output_folder:
    :return:
    """
    relevent_paths = {}
    for root, dirs, files in os.walk(root_dir, topdown=False):
        if "blendshapes.pt" in files:
            action = root.split("/")[-1]
            path = os.path.join('dataset',root.split("dataset/")[-1])
            if action not in relevent_paths.keys():
                relevent_paths[action] = []
            relevent_paths[action].append(path)

    for action in relevent_paths.keys():
        relevent_paths[action].sort()
        print(f'{action}: {len(relevent_paths[action])}')

    import json
    with open(os.path.join(output_folder, f'{dataset_name}_data.json'), 'w') as f:
        json.dump(relevent_paths, f)


def split_data_to_train_and_test(input_json, output_dir, dataset_name):
    import json
    import random

    # Load the original JSON file
    with open(input_json, 'r') as file:
        original_data = json.load(file)

    # Initialize the train and test dictionaries

    if dataset_name=='Express4D':
        train_data = {}
        test_data = {}
        # data_length = len(original_data)
        paths = original_data.keys()
        
        test_paths = random.sample(paths, k=int(len(paths) * 0.2))
        for path in test_paths:
            test_data[path] = original_data[path]
       
        train_paths = [path for path in paths if path not in test_paths]
        for path in train_paths:
            train_data[path] = original_data[path]

    else:
        train_data = []
        test_data = []

        # Iterate over each action in the original data
        for action, paths in original_data.items():
            # Randomly select 20% of the paths
            test_paths = random.sample(paths, k=int(len(paths) * 0.2))
            train_paths = [path for path in paths if path not in test_paths]

            # Assign the selected paths to train and test dictionaries
            train_data.extend(train_paths)
            test_data.extend(test_paths)


    # Create a new JSON file with train and test data
    output_data = {
        "train": train_data,
        "test": test_data
    }

    # Save the new JSON file
    with open(os.path.join(output_dir, f'{dataset_name}_data_train_test_split.json'), 'w') as file:
        json.dump(output_data, file, indent=4)

if __name__== '__main__':
    create_json = False
    create_json_test_train = True
    dataset = 'express4d'
    base_dir = '/home/dcor/yaronaloni/Express4D/dataset/'
    if dataset=='famos':
        root_dir = os.path.join(base_dir, 'FAMOS')
    elif dataset=='coma':
        root_dir = os.path.join(base_dir, 'COMA_data')
    elif dataset=='express4d':
        root_dir = os.path.join(base_dir, 'Express4D')

    if create_json:
        # """data mode is one of the following:
        # landmarks_68, landmarks_68_centralized, landmarks_70, landmarks_70_centralized, blendshapes
        # """
        # data_mode = 'landmarks_70_centralized' # in order to take only once each folder
        # data_mode = 'blendmarks_70_centralized' # in order to take only once each folder
        # data_mode = 'blendshapes' # in order to take only once each folder
        # if famos:
        #     output_folder = os.path.join(base_dir, 'FAMOS_data.json')
        # elif coma:
        #     output_folder = 'dataset/COMA_only_bs_and_lmks'
        generate_json_of_data(root_dir, dataset, base_dir)

    if create_json_test_train:
        if dataset=='famos':
            FaMoS_json = os.path.join(base_dir, 'FAMOS_data.json')
            split_data_to_train_and_test(FaMoS_json, base_dir, 'FAMOS')
        if dataset=='coma':
            COMA_json = os.path.join(base_dir, 'COMA_data.json')
            split_data_to_train_and_test(COMA_json, base_dir, 'COMA')
        if dataset=='express4d':
            Express4D_json = os.path.join(base_dir, 'Express4D_data.json')
            split_data_to_train_and_test(Express4D_json, base_dir, 'Express4D')

def get_data_by_key(json_file, key):
    with open(json_file, 'r') as file:
        data = json.load(file)
        if key in data:
            return data[key]
        else:
            return None
