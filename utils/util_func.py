# built-in
import json
from typing import Tuple

# 3rd party
import numpy as np
from torch.utils.data import Dataset
import torch
from pyfiglet import Figlet
from sklearn.preprocessing import MinMaxScaler

# "cgm_meals_breakfastlunch.json", "img_meals112.json", "demographics-microbiome-data.json"

def make_figlet(text: str, font="smslant", to_print: bool=True):
    f = Figlet(font=font)
    text = f.renderText(text)
    if to_print:
        print(text)
    else:
        return text

# load data from files
def load_data_from_json(
    cgm_path: str|None=None,
    img_path: str|None=None,
    demo_viome_path: str|None=None
) -> Tuple[dict|None, dict|None, dict|None]:
    """
    loads CGM, IMG, and DEMOGRAPHIC/VIOME data, as specified

    Args:
        cgm_path (str, optional): path of cgm reading. Defaults to None if no data to load.
        img_path (str, optional): path of img reading. Defaults to None if no data to load.
        demo_viome_path (str, optional): path of demographic/viome data. Defaults to None if no data to load.

    Returns:
        cgm_meals (dict): cgm readings in dictionary format
        img_meals (dict): img readings in dictionary format
    """

    cgm_data = None if cgm_path == None else cgm_data_loader(cgm_path)
    img_data = None if img_path == None else img_data_loader(img_path)
    demo_viome_data = None if demo_viome_path == None else demo_viome_data_loader(demo_viome_path)

    return cgm_data, img_data, demo_viome_data

def cgm_data_loader(cgm_path: str):
    with open(cgm_path) as json_file:
        cgm_meals = json.load(json_file)
    return cgm_meals

def img_data_loader(img_path: str):
    with open(img_path) as json_file:
        img_meals = json.load(json_file)
    return img_meals

def demo_viome_data_loader(demo_viome_path: str):
    with open(demo_viome_path) as json_file:
        demo_viome = json.load(json_file)

    # Perform data processing here
    # 1. Age (float to float)
    # Create a MinMaxScaler instance for Age
    age_scaler = MinMaxScaler()

    # Extract the "Age" values and reshape them to a 2D array
    ages = [[data["Age"]] for data in demo_viome.values() if "Age" in data]

    # Fit the scaler to the data to compute min and max values
    age_scaler.fit(ages)

    # Transform and update the "Age" values with the scaled values
    for user_id, data in demo_viome.items():
        if "Age" in data:
            age = data["Age"]
            # Reshape the age value to a 2D array before scaling
            scaled_age = age_scaler.transform([[age]])[0][0]
            data["Age"] = scaled_age

    # 2. Gender (str to binary)
    for user_id, data in demo_viome.items():
        # 0 for "Male" and 1 for "Female"
        if "Gender" in data:
            gender = data["Gender"]
            if gender == "M":
                data["Gender"] = 0
            elif gender == "F":
                data["Gender"] = 1

    # 3. BMI = 703 * weight(lb) / height^2(in)
    # Calculate BMI for each row
    for user_id, data in demo_viome.items():
        if "Body weight" in data and "Height" in data:
            weight_lb = data["Body weight"]
            height_inches = data["Height"]
            data["BMI"] = 703 * weight_lb / (height_inches ** 2)

    # Create a MinMaxScaler instance for BMI
    scaler_bmi = MinMaxScaler()

    # Extract the "BMI" values and reshape them to a 2D array
    bmi_values = [[data["BMI"]] for data in demo_viome.values() if "BMI" in data]

    # Fit the scaler to the data to compute min and max values for BMI
    scaler_bmi.fit(bmi_values)

    # Transform and update the "BMI" values with the scaled values
    for user_id, data in demo_viome.items():
        if "BMI" in data:
            bmi = data["BMI"]
            # Reshape the BMI value to a 2D array before scaling
            scaled_bmi = scaler_bmi.transform([[bmi]])[0][0]
            data["BMI"] = scaled_bmi

    # 4. A1c PDL (float to float)
    # Create a MinMaxScaler instance for A1c PDL (Lab)
    scaler_a1c = MinMaxScaler()

    # Extract the "A1c PDL (Lab)" values and reshape them to a 2D array
    a1c_values = [[data["A1c PDL (Lab)"]] for data in demo_viome.values() if "A1c PDL (Lab)" in data]

    # Fit the scaler to the data to compute min and max values for A1c PDL (Lab)
    scaler_a1c.fit(a1c_values)

    # Transform and update the "A1c PDL (Lab)" values with the scaled values
    for user_id, data in demo_viome.items():
        if "A1c PDL (Lab)" in data:
            a1c = data["A1c PDL (Lab)"]
            # Reshape the A1c value to a 2D array before scaling
            scaled_a1c = scaler_a1c.transform([[a1c]])[0][0]
            data["A1c PDL (Lab)"] = scaled_a1c

    # 5. Fasting GLU (float to float)
    # Create a MinMaxScaler instance for Fasting GLU - PDL (Lab)
    scaler_glu = MinMaxScaler()

    # Extract the "Fasting GLU - PDL (Lab)" values and reshape them to a 2D array
    glu_values = [[data["Fasting GLU - PDL (Lab)"]] for data in demo_viome.values() if
                    "Fasting GLU - PDL (Lab)" in data]

    # Fit the scaler to the data to compute min and max values for Fasting GLU - PDL (Lab)
    scaler_glu.fit(glu_values)

    # Transform and update the "Fasting GLU - PDL (Lab)" values with the scaled values
    for user_id, data in demo_viome.items():
        if "Fasting GLU - PDL (Lab)" in data:
            glu = data["Fasting GLU - PDL (Lab)"]
            # Reshape the Fasting GLU value to a 2D array before scaling
            scaled_glu = scaler_glu.transform([[glu]])[0][0]
            data["Fasting GLU - PDL (Lab)"] = scaled_glu

    # 6. Insulin (str to float)
    # Create a MinMaxScaler instance for Insulin
    scaler_insulin = MinMaxScaler()

    # Extract the "Insulin" values and reshape them to a 2D array
    insulin_values = [[data["Insulin"]] for data in demo_viome.values() if "Insulin" in data]

    # Fit the scaler to the data to compute min and max values for Insulin
    scaler_insulin.fit(insulin_values)

    # Transform and update the "Insulin" values with the scaled values
    for user_id, data in demo_viome.items():
        if "Insulin" in data:
            insulin = data["Insulin"]
            # Reshape the Insulin value to a 2D array before scaling
            scaled_insulin = scaler_insulin.transform([[insulin]])[0][0]
            data["Insulin"] = scaled_insulin

    # 7. Top 6 Bacteria (binary to list in binary)
    # "Tannerella sp. 6_1_58FAA_CT1", (missing from data)
    top_6 = ["Alistipes onderdonkii", "Clostridiales bacterium VE202-18", "Filifactor alocis ATCC 35896",
                "Lachnospiraceae bacterium 3-1", "Bifidobacterium adolescentis strain BBMN23",
                "Coprococcus sp. HPP0048"]
    # Iterate over participants in the JSON data
    for user_id, data in demo_viome.items():
        # Create a dict of binary values for the selected bacteria
        bacteria_list = {bacteria: data[bacteria] for bacteria in top_6}
        # Add the dict to the participant's data
        data['Top_6_Bacteria_List'] = bacteria_list

    return demo_viome

# Splits meal_ids into test/train sets pseudo-randomly
def get_train_test_meals(
    cgm_meals: dict, img_meals: dict, test_ratio=0.2, test_random=True, seed=0
) -> tuple[list, list]:
    """
    get_train_test_meals splits and returns the meal_id's into train and test data sets

    Actions:
        1. Split the keys (meal_id's) into train/test depending on the defined ratio and randomization
        2. Return the train and test sets

    Args:
        cgm_meals (dict): dictionary containing CGM readings of meals
        img_meals (dict): dictionary containing image readings of meals
        test_ratio (float, optional): set ratio of test data to be tested. Defaults to 0.2.
        test_random (bool, optional): whether to randomly select test data. Defaults to True.

    Returns:
        train_meals (list): meals selected for training
        test_meals (list): meals selected for testing
    """
    meal_ids = sorted(cgm_meals.keys() & img_meals.keys()) # sorting into same order is necessary for seed to work (since dict keys stored in variable order)

    tol = len(meal_ids)
    test_num = int(tol * test_ratio)

    if test_random:
        np.random.seed(seed)
        test_meals = np.random.choice(meal_ids, test_num) # set seed
    else:
        test_meals = list(range(test_num))

    train_meals = [i for i in meal_ids if i not in test_meals]

    return train_meals, test_meals

# custom Pytorch data loader
class multimodal_dataset(Dataset):
    """ CGM + Image + Demographics + Viome dataset."""

    def __init__(
        self,
        partition, # TODO: is this just name???
        cgm_meals: None|dict = None,
        img_meals: None|dict = None,
        demo_viome_data: None|dict = None, # Add demographics data as an argument
        train_names=None,
        test_names=None,
        batch_size=16,
        drop_last=False,
        random=False,
        test_ratio=0.2,
        normalizer=None,
    ):
        self.bz = batch_size
        self.drop_last = drop_last
        self.max_len = 180 # TODO: should this be hard coded or a tunable parameter?

        self.img_data, self.cgm_data, self.auc_data = [], [], []
        (
            self.race,
            self.age,
            self.gender,
            self.bmi,
            self.a1c,
            self.glu,
            self.insulin,
            self.top10
        ) = ([], [], [], [], [], [], [], [])
        
        (
            self.calorie_label,
            self.labels2,
            self.carb_label,
            self.protein_label,
            self.fat_label,
            self.fiber_label,
        ) = ([], [], [], [], [], [])

        if partition == "train":
            names = train_names
        else:
            names = test_names

        for n in names:
            if (
                cgm_meals[n]["protein"] + cgm_meals[n]["fat"] + cgm_meals[n]["fiber"]
                == 0 # TODO: what if float? Equality might be too strong. Could switch to <0.01?
            ):
                continue

            # Extract demographics data based on the user ID
            user_id = n.split("_")[0]
            demographics_info = self.demo_viome_data.get(user_id, {})  # Replace {} with a default value if necessary

            # Store demographics data as needed
            self.race.append(demographics_info.get("Race", -1))
            self.age.append(demographics_info.get("Age", -1))
            self.gender.append(demographics_info.get("Gender", -1))
            self.bmi.append(demographics_info.get("BMI", -1))
            self.a1c.append(demographics_info.get("A1c PDL (Lab)", -1))
            self.glu.append(demographics_info.get("Fasting GLU - PDL (Lab)", -1))
            self.insulin.append(demographics_info.get("Insulin - PDL (Lab)", -1))
            self.top10.append(demographics_info.get("Top 6 Bacteria", -1))

            # TODO: stopped here
            # self.cgm_data.append(self.padding([cgm_meals[n]['dexcom'], cgm_meals[n]['libre']]))
            raw_data1 = self.padding([cgm_meals[n]["libre_kalman"]])
            # raw_data2 = self.padding([cgm_meals[n]['libre_kalman']])
            # auc_data1 = self.get_gAUC_5(raw_data1-raw_data1[0]*np.ones_like(raw_data1))
            # auc_data2 = self.get_gAUC_5(raw_data2-raw_data2[0]*np.ones_like(raw_data2))
            auc_data1 = self.get_gAUC_5(raw_data1)
            # auc_data2 = self.get_gAUC_5(raw_data2)
            self.cgm_data.append(np.expand_dims(raw_data1, -1))
            self.auc_data.append(auc_data1)
            self.img_data.append(img_meals[n][0])
            self.calorie_label.append(
                cgm_meals[n]["calories"]
            )  # cgm_meals[n]['calories']
            # print(cgm_meals[n]['protein']+cgm_meals[n]['fat']+cgm_meals[n]['fiber'])
            self.labels2.append(
                cgm_meals[n]["carbs"]
                / (
                    cgm_meals[n]["protein"]
                    + cgm_meals[n]["fat"]
                    + cgm_meals[n]["fiber"]
                ) # calculates the ratio of carbohydrates to the sum of protein, fat, and fiber for the current meal
            )
            self.carb_label.append(cgm_meals[n]["carbs"])
            self.protein_label.append(cgm_meals[n]["protein"])
            self.fat_label.append(cgm_meals[n]["fat"])
            self.fiber_label.append(cgm_meals[n]["fiber"])
        print(len(self.calorie_label), len(self.carb_label))

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        image = self.img_data[idx]
        signal = self.cgm_data[idx]
        label = self.labels[idx]
        if self.transform: # FIXME: TODO: the transform function is not defined
            image = self.transform(image)

        return (
            torch.from_numpy(np.array(image)).float(),
            torch.from_numpy(np.array(signal)).float(),
            torch.from_numpy(np.array([label])).float(),
        )

    def padding(self, signal):
        num_fea = len(signal)
        length = len(signal[0])
        if length > self.max_len:
            data = np.array(signal)[:, :self.max_len]
        if length <= self.max_len:
            # print(222, np.shape(list(np.concatenate((signal, np.zeros((2, max_len-length))), -1))))
            data = np.concatenate(
                (signal, np.zeros((num_fea, self.max_len - length))), -1
            )
        return list(np.squeeze(np.transpose(data)))

    def normal_function(self, mu, sigma, x):
        normal_function_value = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((x - mu) ** 2) / (2 * (sigma**2))
        )
        return normal_function_value

    def get_gAUC_5(self, data):
        length = len(data)
        num_kernels = 5
        kernel_len = round(length / num_kernels)
        sigma = kernel_len / 1.96
        temp = []
        temp.append(
            np.trapz(
                self.normal_function(0, sigma, np.asarray([i for i in range(0, 45)]))
                * data[0:45]
            )
        )
        temp.append(
            np.trapz(
                self.normal_function(45, sigma, np.asarray([i for i in range(0, 90)]))
                * data[0:90]
            )
        )
        temp.append(
            np.trapz(
                self.normal_function(90, sigma, np.asarray([i for i in range(45, 135)]))
                * data[45:135]
            )
        )
        temp.append(
            np.trapz(
                self.normal_function(
                    135, sigma, np.asarray([i for i in range(90, 180)])
                )
                * data[90:180]
            )
        )
        temp.append(
            np.trapz(
                self.normal_function(
                    180, sigma, np.asarray([i for i in range(135, 180)])
                )
                * data[135:180]
            )
        )
        temp = np.array([i for i in temp])  # .reshape(-1, 1)
        return temp

    def get_gAUC_3(self, data):
        length = len(data)
        p0, p1, p2 = 0, length // 2, length
        num_kernels = 5
        kernel_len = round(length / num_kernels)
        sigma = kernel_len / 1.96
        temp = []
        temp.append(
            np.trapz(
                self.normal_function(p0, sigma, np.asarray([i for i in range(p0, p1)]))
                * data[p0:p1]
            )
        )
        temp.append(
            np.trapz(
                self.normal_function(p1, sigma, np.asarray([i for i in range(p0, p2)]))
                * data[p0:p2]
            )
        )
        temp.append(
            np.trapz(
                self.normal_function(
                    p2 - 1, sigma, np.asarray([i for i in range(p1, p2)])
                )
                * data[p1:p2]
            )
        )
        temp = np.array([i for i in temp])  # .reshape(-1, 1)
        return temp

    def __iter__(self):
        length = len(self.img_data)
        num = length // self.bz
        for i in range(num + 1):
            if i < num:
                image = self.img_data[i * self.bz : (i + 1) * self.bz]
                signal = self.cgm_data[i * self.bz : (i + 1) * self.bz]
                auc = self.auc_data[i * self.bz : (i + 1) * self.bz]
                calorie_label = self.calorie_label[i * self.bz : (i + 1) * self.bz]
                labels2 = self.labels2[i * self.bz : (i + 1) * self.bz]
                carb_label = self.carb_label[i * self.bz : (i + 1) * self.bz]
                protein_label = self.protein_label[i * self.bz : (i + 1) * self.bz]
                fat_label = self.fat_label[i * self.bz : (i + 1) * self.bz]
                fiber_label = self.fiber_label[i * self.bz : (i + 1) * self.bz]
            else:
                image = self.img_data[i * self.bz :]
                signal = self.cgm_data[i * self.bz :]
                auc = self.auc_data[i * self.bz :]
                calorie_label = self.calorie_label[i * self.bz :]
                labels2 = self.labels2[i * self.bz :]
                carb_label = self.carb_label[i * self.bz :]
                protein_label = self.protein_label[i * self.bz :]
                fat_label = self.fat_label[i * self.bz :]
                fiber_label = self.fiber_label[i * self.bz :]
            if len(image) == self.bz:
                yield (
                    torch.from_numpy(np.array(image)).float(),
                    torch.from_numpy(np.array(signal)).float(),
                    torch.from_numpy(np.array(auc)).float(),
                    torch.from_numpy(np.array(calorie_label)).float(),
                    torch.from_numpy(np.array(labels2)).float(),
                    torch.from_numpy(np.array(carb_label)).float(),
                    torch.from_numpy(np.array(protein_label)).float(),
                    torch.from_numpy(np.array(fat_label)).float(),
                    torch.from_numpy(np.array(fiber_label)).float(),
                )
            elif len(image) > 0 and not self.drop_last:
                yield (
                    torch.from_numpy(np.array(image)).float(),
                    torch.from_numpy(np.array(signal)).float(),
                    torch.from_numpy(np.array(auc)).float(),
                    torch.from_numpy(np.array(calorie_label)).float(),
                    torch.from_numpy(np.array(labels2)).float(),
                    torch.from_numpy(np.array(carb_label)).float(),
                    torch.from_numpy(np.array(protein_label)).float(),
                    torch.from_numpy(np.array(fat_label)).float(),
                    torch.from_numpy(np.array(fiber_label)).float(),
                )

    def collate(self, batch):
        (image, signal, labels) = zip(*batch)
        return [image, signal, labels]

# def add_demo_viome_to_meals():
#     '''
#     creates a dataset with meals with corresponding demographics and gut microbiome data

#     Returns:
#         cgm_img_viome_meals (dict): cgm, image, demographics, and viome readings in dictionary format
#     '''
#     # Load CGM and image data
#     cgm_meals, img_meals = load_cgm_and_img_from_json(cgm_path="cgm_meals_breakfastlunch.json",
#                                                       img_path="img_meals112.json")

#     # Load demographics and viome data
#     demo = load_demo_viome_from_json(demo_viome_path="demographics-microbiome-data.json")

#     # Extract the common meals (keys) between CGM and image data
#     cgm_meals = list(cgm_meals.keys())
#     img_meals = list(img_meals.keys())
#     meals = list(set(cgm_meals) & set(img_meals))

#     # Create a new dictionary to store combined data
#     cgm_img_viome_meals = {}

#     # Iterate through each common meal
#     # Keys for cgm & img = (user_id)_(meal#) (ex. 1001_1)
#     # Keys for demo = (user_id) (ex. 1001)
#     for meal in meals:
#         # Extract user_id from the meal key
#         user_id = meal.split('_')[0]

#         # Check if the user_id exists in the demographics data
#         if user_id in demo:
#             # Create a new dictionary entry for this meal
#             cgm_img_viome_meals[meal] = {
#                 'cgm_data': cgm_meals[meal],
#                 'img_data': img_meals[meal],
#                 'demo_data': demo[user_id]
#             }

#     # Return the combined dataset
#     return cgm_img_viome_meals