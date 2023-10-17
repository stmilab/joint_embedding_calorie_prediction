import argparse


def str2bool(v: str) -> bool:
    """
    str2bool
        Convert String into Boolean

    Arguments:
        v {str} --input raw string of boolean command

    Returns:
        bool -- converted boolean value
    """
    return v.lower() in ("yes", "true", "y", "t", "1")


def sicong_argparse(model: str) -> argparse.Namespace:
    """
    sicong_argparse
        parsing command line arguments with reinforced formats

    Arguments:
        model {str} -- indicates which model being used

    Returns:
        argparse.Namespace -- flags containining the specification of this run
    """
    model_desc_dict = {
        "multi-modal": "Using multi-modal to predict macro nutrient",
        "cgm-only": "only using CGM readings to predict macro nutrient",
    }
    if model not in model_desc_dict:
        raise RuntimeError(
            "Model Unknown, only 'Sequnet' and 'Transformer' are available at this time"
        )
    parser = argparse.ArgumentParser(description=model_desc_dict[model])
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Choosing the Batch Size, default=32 (If out of memory, try a smaller batch_size)",
    )
    parser.add_argument(
        "--cgm_backbone",
        default="transformer",
        help="Select the backbone to extract CGM data (defaults to transformer)",
    )
    parser.add_argument(
        "--img_backbone",
        default="ViT",
        help="Select the backbone to extract image data (defaults to vision transformer)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="number of attention heads, alternatively, number of layers, default to 4",
    )
    parser.add_argument(
        "--num_dim",
        type=int,
        default=32,
        help="number of LSTM dimension, or number of transformer classes default=32",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=112,
        help="size of image input default=112",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Weight Decay hyperparameter default=0",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Choose the max number of epochs, default=3 for testing purpose",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Define Learning Rate, default=1e-4, if failed try something smaller",
    )
    parser.add_argument(
        "--ignore_first_meal",
        type=str2bool,
        default=False,
        help="Whether to ignore the first meal in the sequence (usually the first meal is a calibration meal))",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0,
        help="Define the dropout rate, default=0",
    )
    parser.add_argument("--save_result", type=str2bool, default=False)
    parser.add_argument(
        "--sel_gpu",
        type=int,
        default=1,
        help="Choosing which GPU to use (STMI has GPU 0~7)",
    )
    parser.add_argument(
        "--shuffle_data",
        type=str2bool,
        default=False,
        help="Whether to shuffle data before train/test split",
    )
    parser.add_argument(
        "--use_wandb",
        type=str2bool,
        default=False,
        help="Whether to save progress and results to Wandb",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="BSN-MacroNutrient",
        help="If using Wandb, this shows the location of the project, usually don't change this one",
    )
    parser.add_argument(
        "--wandb_tag",
        type=str,
        default="default_sequnet",
        help="If using Wandb, define tag to help filter results",
    )
    flags, _ = parser.parse_known_args()
    # setting cuda device
    flags.device = f"cuda:{flags.sel_gpu}" if flags.sel_gpu > 0 else "cpu"
    print("Flags:")
    for k, v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))
    return flags


if __name__ == "__main__":
    flags = sicong_argparse("Transformer")
    print("This main func is used for testing purpose only")
