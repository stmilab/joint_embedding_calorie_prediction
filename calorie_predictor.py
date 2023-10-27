from utils import arg_parser, util_func as ut
from models import (
    lstm,
    other_layers as ol,
    vision_transformer as vit,
    transformer as tsfm,
)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb
import os
from tqdm import tqdm
import wandb
import torchvision.models as vision_models

def load_train_test(flags: arg_parser):
    """
    load_train_test loading trnd and test datasets

    This function loads datasets from JSON files or pre-saved pytorch files

    Args:
        flags (arg_parser): arguments for the model

    Returns:
        train_loader: torch loader for training data
        test_loader: torch loader for testing data
    """
    # Creating dataloaders with train and test
    if os.path.exists("train_dataset.pt"):
        ut.make_figlet("loading datasets from saved pytorch files")
        train_dataset = torch.load("train_dataset.pt")
        test_dataset = torch.load("test_dataset.pt")
    else:
        # Loading CGM and IMG data from JSON files
        cgm_meals, img_meals, demo_viome = ut.load_data_from_json("cgm_meals_breakfastlunch.json", "img_meals112.json", "demographics-microbiome-data.json")

        # Split into train and test meals
        train_meals, test_meals = ut.get_train_test_meals(cgm_meals, img_meals, test_ratio=0.2)
        
        # create Pytorch data loaders
        train_dataset = ut.multimodal_dataset("train", cgm_meals, img_meals, demo_viome, train_meals)
        test_dataset = ut.multimodal_dataset("test", cgm_meals, img_meals, demo_viome, None, test_meals)

        # cache for future computation
        torch.save(train_dataset, "train_dataset.pt")
        torch.save(test_dataset, "test_dataset.pt")

    train_dataset.batch_size = flags.batch_size
    return train_dataset, test_dataset

def declare_multimodal_model(
    cgm_backbone: str,
    img_backbone: str,
    prediction_tasks: list,
    flags: arg_parser,
):
    """
    declare_multimodal_model declares the multimodal model for matronutrient prediction task

    _extended_summary_

    Args:
        cgm_backbone (str): backbone selected for cgm model
        img_backbone (str): backbone selected for image model
        prediction_tasks (list): list of prediction tasks, e.g. ["carbs", "fat", "protein"]
        flags (arg_parser): arguments for the model
    """
    print(f"Using {cgm_backbone} as cgm backbone; {img_backbone} as image backbone")
    # Start of cgm model
    if cgm_backbone.lower() == "lstm":
        cgm_model = lstm.LSTM(
            input_size=1,
            hidden_size=flags.num_dim,
            dropout=flags.dropout_rate,
            device=flags.device,
        ).to(flags.device)
        model_kwargs = (
            flags.img_size,
            flags.num_dim,
            3,
            2,
            2,
            flags.num_dim,
            4,
            (flags.img_size // 4) ** 2,
            0.2,
        )
    elif cgm_backbone.lower() == "transformer":
        cgm_model = tsfm.MultiheadAttention(
            n_features=1,
            embed_dim=96,
            num_heads=4,
            num_classes=64,
            dropout=0.2,
            num_layers=6,
        ).to(flags.device)
        model_kwargs = (112, 64, 3, 4, 4, 64, 4, (112 // 4) ** 2)
    else:
        raise NotImplementedError("Only support LSTM and Transformer as cgm backbone!")
    # Start of image model
    if img_backbone.lower() == "vit":
        img_model = vit.ViT(model_kwargs, train_loader).to(flags.device)
    elif img_backbone.lower() == "vgg":
        img_model = vision_models.vgg19(pretrained=True).to(flags.device)
    elif img_backbone.lower() == "resnet":
        img_model = vision_models.resnet50(pretrained=True).to(flags.device)
    else:
        raise NotImplementedError("Currently only support ViT, VGG< and ResNET as image backbone!")
    # Combining the multimodal model
    multimodal_model = ol.MultiModalModel(
        [img_model, cgm_model],
    ).to(flags.device)

    # Start of prediction model
    if len(prediction_tasks) == 1:
        nutrient_predictor = ol.Regressor(64 * 2 + 5).to(flags.device)
    optimizer = optim.Adam(
        list(nutrient_predictor.parameters())
        + list(img_model.parameters())
        + list(cgm_model.parameters()),
        lr=flags.lr,
        weight_decay=flags.weight_decay,
    )
    return multimodal_model, nutrient_predictor, optimizer

if __name__ == "__main__":
    # defining the specifications used for this RUN
    flags = arg_parser.sicong_argparse("multi-modal")
    # declaring weights & biases logging
    if flags.use_wandb:
        wandb.init(
            project=flags.wandb_project,
            reinit=True,
            tags=[flags.wandb_tag],
        )
        wandb.config.update(flags)
    log_dict = {}
    # loading data
    train_loader, test_loader = load_train_test(flags)
    # declaring model
    multimodal_model, nutrient_predictor, optimizer = declare_multimodal_model(
        cgm_backbone=flags.cgm_backbone,
        img_backbone=flags.img_backbone,
        prediction_tasks=["carbs"],
        flags=flags,
    )
    # pdb.set_trace()

    # Lida's addition
    min_loss = float("inf")  # the minimum calculated loss
    calorie_label_train = torch.mean(torch.Tensor(train_loader.calorie_label)).to(
        flags.device
    )
    carb_label_train = torch.mean(torch.Tensor(train_loader.carb_label)).to(
        flags.device
    )
    ut.make_figlet("training")
    for epoch in range(flags.epochs):
        epoch_train_loss = []
        for idx, (
            img_data,
            cgm_data,
            auc_data,
            calorie_label,
            label2,
            carb_label,
            protein_label,
            fat_label,
            fiber_label,
        ) in enumerate(train_loader):
            optimizer.zero_grad()
            img_tensor = img_data
            cgm_tensor = cgm_data
            carb_label = torch.unsqueeze(carb_label, -1).to(flags.device)
            img_embedding, cgm_embedding = multimodal_model(
                [img_tensor.to(flags.device), cgm_tensor.to(flags.device)]
            )

            data = torch.cat(
                (img_embedding, cgm_embedding, auc_data.to(flags.device)), -1
            )
            carb_pred = nutrient_predictor(data)
            # calculate rmsre loss
            msre = torch.mean((carb_pred - carb_label) ** 2 / carb_label**2)
            msre.backward()
            optimizer.step()
            with torch.no_grad():
                epoch_train_loss.append(msre.cpu().numpy())
        if epoch % 10 == 0:
            print(f"epoch={epoch}, Train Loss={np.array(epoch_train_loss).mean():.3f}")
        if flags.use_wandb:
            wandb.log({"train_progress_rmsre": np.array(epoch_train_loss).mean()})
        epoch_total_loss = []
        for _, (
            img_data,
            cgm_data,
            auc_data,
            calorie_label,
            label2,
            carb_label,
            protein_label,
            fat_label,
            fiber_label,
        ) in enumerate(test_loader):
            img_tensor = img_data
            cgm_tensor = cgm_data
            calorie_label = torch.unsqueeze(calorie_label, -1)
            carb_label = torch.unsqueeze(carb_label, -1)

            img_embedding, cgm_embedding = multimodal_model(
                [img_tensor.to(flags.device), cgm_tensor.to(flags.device)]
            )

            merged_embedding = torch.cat(
                (img_embedding, cgm_embedding, auc_data.to(flags.device)), -1
            )
            carb_pred = nutrient_predictor(merged_embedding)
            # rmsre loss calculation
            with torch.no_grad():
                carb_loss = ((carb_label - carb_pred.cpu()) / (carb_label)) ** 2
            epoch_total_loss.extend(list(carb_loss))
        carb_rmsre = np.mean(np.array([i[0] for i in epoch_total_loss])) ** 0.5
        updated = ""
        if carb_rmsre < min_loss:
            min_loss = carb_rmsre
            updated = "true"
            print(f"epoch={epoch}, predicted carbs={carb_rmsre:.3f}, updated={updated}")
        if flags.use_wandb:
            wandb.log({"test_progress_rmsre": carb_rmsre})
    if flags.use_wandb:
        wandb.log({"min_rmsre": min_loss})
    print(f"min_loss={min_loss}")
