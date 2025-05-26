import json
import os
from post_processing.clean_data import interpolate_series
from post_processing.filters import butterworth_filter, gaussian_filter
from utils.video_info import extract_video_info
from utils import config
from utils.joint_enum import PredJoints
from utils.utils import plot_filtering_effect


def save_filtered_keypoints(
    output_folder, original_json_path, filtered_keypoints, filter_name
):
    os.makedirs(output_folder, exist_ok=True)
    filtered_json_path = os.path.join(
        output_folder,
        os.path.basename(original_json_path).replace(
            ".json", f"_{filter_name}_filtered.json"
        ),
    )
    with open(filtered_json_path, "w") as f:
        json.dump(filtered_keypoints, f, indent=4)
    print(f"Filtered keypoints saved to: {filtered_json_path}")


def run_filter(filter_name, filter_kwargs):
    base_path = config.VIDEO_FOLDER
    output_base = (
        r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_x_degraded_40"
    )
    iqr_multiplier = 1.5
    interpolation_kind = "linear"
    joints = [
        PredJoints.LEFT_ANKLE.value,
        PredJoints.RIGHT_ANKLE.value,
        PredJoints.LEFT_HIP.value,
        PredJoints.RIGHT_HIP.value,
        PredJoints.LEFT_KNEE.value,
        PredJoints.RIGHT_KNEE.value,
    ]

    for root, dirs, files in os.walk(base_path):
        for file in files:
            video_info = extract_video_info(file, root)
            if not video_info:
                continue

            subject, action, camera = video_info
            action_group = action.replace(" ", "_")
            json_path = os.path.join(
                output_base,
                subject,
                f"{action_group}_({'C' + str(camera + 1)})",
                f"{action_group}_({'C' + str(camera + 1)})/{action_group}_({'C' + str(camera + 1)})".replace(
                    " ", ""
                )
                + ".json",
            )

            if not os.path.exists(json_path):
                print(f"File not found: {json_path}")
                continue

            with open(json_path, "r") as f:
                pred_keypoints = json.load(f)

            for kp_set in range(len(pred_keypoints[0]["keypoints"][0]["keypoints"])):
                for joint in joints:
                    x_series, y_series = [], []
                    for frame in pred_keypoints:
                        kp = frame["keypoints"][0]["keypoints"][kp_set][joint]
                        x_series.append(kp[0])
                        y_series.append(kp[1])

                    try:
                        x_interp = interpolate_series(
                            x_series, iqr_multiplier, interpolation_kind
                        )
                        y_interp = interpolate_series(
                            y_series, iqr_multiplier, interpolation_kind
                        )

                        if filter_name == "gaussian":
                            x_filtered = gaussian_filter(x_interp, **filter_kwargs)
                            y_filtered = gaussian_filter(y_interp, **filter_kwargs)
                        elif filter_name == "butterworth":
                            x_filtered = butterworth_filter(x_interp, **filter_kwargs)
                            y_filtered = butterworth_filter(y_interp, **filter_kwargs)
                        else:
                            raise ValueError(f"Unknown filter: {filter_name}")
                    except Exception as e:
                        print(f"Error filtering joint {joint}: {e}")
                        continue

                    if joint == PredJoints.LEFT_ANKLE.value and kp_set == 0:
                        plot_dir = os.path.join(
                            output_base,
                            subject,
                            f"{action_group}_({'C' + str(camera + 1)})",
                            "plots",
                        )
                        os.makedirs(plot_dir, exist_ok=True)

                        plot_filtering_effect(
                            original=x_series,
                            filtered=x_filtered,
                            title=f"X - {subject} {action} Joint {joint}",
                            save_path=os.path.join(
                                plot_dir, f"x_{joint}_{filter_name}.png"
                            ),
                        )
                        plot_filtering_effect(
                            original=y_series,
                            filtered=y_filtered,
                            title=f"Y - {subject} {action} Joint {joint}",
                            save_path=os.path.join(
                                plot_dir, f"y_{joint}_{filter_name}.png"
                            ),
                        )

                    for i, frame in enumerate(pred_keypoints):
                        frame["keypoints"][0]["keypoints"][kp_set][joint][0] = float(
                            x_filtered[i]
                        )
                        frame["keypoints"][0]["keypoints"][kp_set][joint][1] = float(
                            y_filtered[i]
                        )

            output_folder = os.path.join(
                output_base,
                subject,
                f"{action_group}_({'C' + str(camera + 1)})",
                filter_name,
            )
            save_filtered_keypoints(
                output_folder, json_path, pred_keypoints, filter_name
            )


if __name__ == "__main__":
    run_filter("gaussian", {"sigma": 1})
