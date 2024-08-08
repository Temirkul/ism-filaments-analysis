# NOTE: this script should be ran from the root of the repository

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
from perlin_numpy import generate_fractal_noise_2d
from tqdm import tqdm


def generate_dataset(
    num_images: int = 10000,
    min_filaments: int = 1,
    max_filaments: int = 10,
    add_perlin_noise: bool = False,
    vmin: float = 0,
    vmax: float = 0,  # this controls how intense the perlin noise is. 0 -- no noise, 1 -- lots of noise
    rect_noise_num: int = 0,
    rect_noise_scale_factor: float = 3.5,
    img_width: int = 256,
    img_height: int = 256,
) -> str:
    output_dir = f"data/Rects/{img_width}_{img_height}_{min_filaments}_{max_filaments}_{vmin}_{vmax}_{rect_noise_num}_{rect_noise_scale_factor}rects_{num_images}"

    if os.path.isdir(output_dir) == False:
        os.makedirs(output_dir)
        os.mkdir(
            f"data/Rects/{img_width}_{img_height}_{min_filaments}_{max_filaments}_{vmin}_{vmax}_{rect_noise_num}_{rect_noise_scale_factor}rects_{num_images}/imgs"
        )
        os.mkdir(
            f"data/Rects/{img_width}_{img_height}_{min_filaments}_{max_filaments}_{vmin}_{vmax}_{rect_noise_num}_{rect_noise_scale_factor}rects_{num_images}/masks"
        )

    for k in tqdm(range(num_images)):
        rotation_point = "xy"

        max_width = img_width / 4
        max_height = img_height / 12

        min_width = img_width / 32
        min_height = img_height / 64

        my_dpi = 96

        fig1, ax1 = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=((img_width + 76) / my_dpi, (img_height + 77) / my_dpi),
            dpi=my_dpi,
        )
        fig2, ax2 = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=((img_width + 75) / my_dpi, (img_height + 77) / my_dpi),
            dpi=my_dpi,
        )
        ax1.set_xlim(0, img_width)
        ax1.set_ylim(0, img_height)
        ax1.invert_yaxis()
        ax1.axis("off")
        ax2.set_xlim(0, img_width)
        ax2.set_ylim(0, img_height)
        ax2.invert_yaxis()
        ax2.axis("off")

        num_filaments = np.random.randint(low=min_filaments, high=max_filaments + 1)

        for i in range(num_filaments):
            if i == 0:
                params = initial_params(
                    img_width, img_height, max_width, max_height, min_width, min_height
                )
                rect = patches.Rectangle(
                    xy=(params[0], params[1]),
                    width=params[2],
                    height=params[3],
                    angle=params[4],
                    rotation_point=rotation_point,  # type: ignore
                    color="w",
                )
                color_mask = get_color_mask(params[4])
                rect_mask = patches.Rectangle(
                    xy=(params[0], params[1]),
                    width=params[2],
                    height=params[3],
                    angle=params[4],
                    rotation_point=rotation_point,  # type: ignore
                    color=color_mask,
                )
                rect.set_antialiased(False)
                rect_mask.set_antialiased(False)
                ax1.add_patch(rect)
                ax2.add_patch(rect_mask)
            else:
                w = np.random.randint(low=min_width, high=max_width)  # type: ignore
                h = np.random.randint(low=min_height, high=max_height)  # type: ignore

                x, y = get_xy(params[0], params[1], params[2], params[4])

                angle = np.random.randint(
                    low=-70, high=70
                )  # try different angle ranges to attempt to avoid cluttering
                params = (x, y, w, h, angle)

                if (
                    x < 0 or y < 0 or x > img_width or y > img_height
                ):  # if out of bounds, create new random filament
                    params = initial_params(
                        img_width,
                        img_height,
                        max_width,
                        max_height,
                        min_width,
                        min_height,
                    )

                rect = patches.Rectangle(
                    xy=(params[0], params[1]),
                    width=params[2],
                    height=params[3],
                    angle=params[4],
                    rotation_point=rotation_point,  # type: ignore
                    color="w",
                    zorder=2,
                )
                color_mask = get_color_mask(params[4])
                rect_mask = patches.Rectangle(
                    xy=(params[0], params[1]),
                    width=params[2],
                    height=params[3],
                    angle=params[4],
                    rotation_point=rotation_point,  # type: ignore
                    color=color_mask,
                )
                rect.set_antialiased(False)
                rect_mask.set_antialiased(False)
                ax1.add_patch(rect)
                ax2.add_patch(rect_mask)

        for i in range(rect_noise_num):
            # noise
            w = np.random.randint(low=min_width * rect_noise_scale_factor, high=max_width * rect_noise_scale_factor)  # type: ignore
            h = np.random.randint(low=min_height * rect_noise_scale_factor, high=max_height * rect_noise_scale_factor)  # type: ignore

            x, y = get_xy(params[0], params[1], params[2], params[4])

            angle = np.random.randint(
                low=-70, high=70
            )  # try different angle ranges to attempt to avoid cluttering
            params = (x, y, w, h, angle)

            if (
                x < 0 or y < 0 or x > img_width or y > img_height
            ):  # if out of bounds, create new random filament
                params = initial_params(
                    img_width,
                    img_height,
                    max_width * rect_noise_scale_factor,
                    max_height * rect_noise_scale_factor,
                    min_width * rect_noise_scale_factor,
                    min_height * rect_noise_scale_factor,
                )
            c = np.random.randint(low=0, high=100)
            color = (
                c / 255,
                c / 255,
                c / 255,
            )  # sets the color to be less visible than white (not as bright as the filaments)
            rect = patches.Rectangle(
                xy=(params[0], params[1]),
                width=params[2],
                height=params[3],
                angle=params[4],
                rotation_point=rotation_point,  # type: ignore
                color=color,
                zorder=1,
            )
            rect.set_antialiased(False)
            ax1.add_patch(rect)

        if add_perlin_noise:
            noise = generate_fractal_noise_2d(
                shape=(img_width, img_height), res=(4, 4), octaves=4
            )
            ax1.imshow(
                noise,
                origin="upper",
                extent=(0, img_width, 0, img_height),
                interpolation="none",
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
            )

        plt.style.use("dark_background")

        fig1.savefig(
            f"data/Rects/{img_width}_{img_height}_{min_filaments}_{max_filaments}_{vmin}_{vmax}_{rect_noise_num}_{rect_noise_scale_factor}rects_{num_images}/imgs/rect{k}_img.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=my_dpi,
        )
        fig2.savefig(
            f"data/Rects/{img_width}_{img_height}_{min_filaments}_{max_filaments}_{vmin}_{vmax}_{rect_noise_num}_{rect_noise_scale_factor}rects_{num_images}/masks/rect{k}_mask.png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=my_dpi,
        )
        plt.close(fig1)
        plt.close(fig2)

    return output_dir


def initial_params(
    img_width, img_height, max_width, max_height, min_width, min_height
):  # initial params of the first rectangle. Rectangles are in XYWHA, where X and Y are the ..., W & H are the widths and heights, and A is the angle
    w = np.random.randint(low=min_width, high=max_width)
    h = np.random.randint(low=min_height, high=max_height)
    x = np.random.randint(low=0, high=img_width - w)
    y = np.random.randint(low=0, high=img_height - h)
    angle = np.random.randint(low=-90, high=90)

    return x, y, w, h, angle


def get_xy(x_prev, y_prev, w_prev, angle_prev):
    if angle_prev > 0 and angle_prev <= 90:
        x = x_prev + w_prev * np.cos(angle_prev * np.pi / 180)
        y = y_prev + w_prev * np.sin(angle_prev * np.pi / 180)
    elif angle_prev > 90 and angle_prev <= 180:
        x = x_prev - w_prev * np.sin(angle_prev * np.pi / 180 - np.pi / 2)
        y = y_prev + w_prev * np.cos(angle_prev * np.pi / 180 - np.pi / 2)
    elif angle_prev > 180 and angle_prev <= 270:
        x = x_prev - w_prev * np.cos(angle_prev * np.pi / 180 - np.pi)
        y = y_prev - w_prev * np.sin(angle_prev * np.pi / 180 - np.pi)
    elif angle_prev > 270 and angle_prev <= 360:
        x = x_prev + w_prev * np.sin(angle_prev * np.pi / 180 - 3 * np.pi / 2)
        y = y_prev - w_prev * np.cos(angle_prev * np.pi / 180 - 3 * np.pi / 2)
    elif angle_prev <= 0 and angle_prev >= -90:
        x = x_prev + w_prev * np.cos(angle_prev * np.pi / 180)
        y = y_prev + w_prev * np.sin(
            angle_prev * np.pi / 180
        )  # + because angle is negative

    return x, y


def get_color_mask(angle):
    if angle == 0:
        color_mask = (90 / 255, 90 / 255, 90 / 255, 1)
    elif angle == 90 or angle == -90:
        color_mask = (180 / 255, 180 / 255, 180 / 255, 1)
    elif angle > 0 and angle < 90:
        color_mask = ((90 - angle) / 255, (90 - angle) / 255, (90 - angle) / 255, 1)
    elif angle < 0 and angle > -90:
        color_mask = (
            (90 - angle) / 255,
            (90 - angle) / 255,
            (90 - angle) / 255,
            1,
        )  # - because angle is negative

    return color_mask


if __name__ == "__main__":
    output_dir = generate_dataset()
    splitfolders.ratio(
        input=output_dir,
        output=f"{output_dir}_output",
        seed=1337,
        ratio=(0.9, 0.1),
        group_prefix=None,
        move=False,
    )
