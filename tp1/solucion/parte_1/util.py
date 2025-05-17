from skimage import io
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


def load_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return io.imread(image_path)


def display_image(image, image_title: str = "Original Image") -> None:
    plt.figure(figsize=(10, 10))
    plt.title(image_title)
    plt.imshow(image)
    plt.show()



COLUMNS = ["Mean", "Std", "Min", "Median", "P_80", "P_90", "P_99", "Max"]


def compute_channel_stats(channel: np.ndarray[np.uint8]) -> list[float]:
    return [
        np.mean(channel),
        np.std(channel),
        np.min(channel),
        np.median(channel),
        np.percentile(channel, 80),
        np.percentile(channel, 90),
        np.percentile(channel, 99),
        np.max(channel),
    ]


def calc_color_overcast(image: np.ndarray[np.uint8]) -> pd.DataFrame:
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    channel_stats = pd.DataFrame(columns=COLUMNS)
    for channel, name in zip(
        [red_channel, green_channel, blue_channel], ["Red", "Green", "Blue"]
    ):
        channel_stats.loc[name] = compute_channel_stats(channel)

    return channel_stats
