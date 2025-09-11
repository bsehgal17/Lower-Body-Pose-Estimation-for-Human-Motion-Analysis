import cv2
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA


def enhance_dark_video_with_labeled_collage(
    input_path,
    output_dir,
    clip_limit=0.5,
    tile_grid_size=(8, 8),
    gamma_values=[1.0, 5.0],
    use_pca=True,  # Choose between PCA or manual fusion
    manual_weight=0.3,  # Weight for manual fusion (0.0 to 1.0)
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    steps = [
        "original",
        "v_inv",
        "v_clahe",
        "gamma1",
        "gamma5",
        "tophat",
        "pca_fused",
        "manual_fused",
        "final_output",
    ]

    writers = {
        step: cv2.VideoWriter(
            str(output_dir / f"{step}.mp4"), fourcc, fps, (width, height)
        )
        for step in steps
    }

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Disk structuring element for top-hat
    radius = 3
    kernel_size = 2 * radius + 1
    structuring_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    frame_count = 0
    collage_images = {}
    step_images = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Store original frame
        writers["original"].write(frame)
        if frame_count == 2:
            collage_images["original"] = frame.copy()
            step_images["original"] = frame.copy()

        # Step 1: Convert RGB to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Step 2: Normalize V component
        v_norm = cv2.normalize(v.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX)

        # Step 3: Inverse of normalized V
        v_inv = 1.0 - v_norm
        v_inv_disp = (v_inv * 255).astype("uint8")
        writers["v_inv"].write(cv2.cvtColor(v_inv_disp, cv2.COLOR_GRAY2BGR))
        if frame_count == 2:
            step_images["v_inv"] = cv2.cvtColor(v_inv_disp, cv2.COLOR_GRAY2BGR)
            collage_images["v_inv"] = step_images["v_inv"].copy()

        # Step 4: Apply CLAHE on inverted V
        v_clahe = clahe.apply(v_inv_disp)  # uint8
        v_clahe_norm = v_clahe.astype("float32") / 255.0
        writers["v_clahe"].write(cv2.cvtColor(v_clahe, cv2.COLOR_GRAY2BGR))
        if frame_count == 2:
            step_images["v_clahe"] = cv2.cvtColor(v_clahe, cv2.COLOR_GRAY2BGR)
            collage_images["v_clahe"] = step_images["v_clahe"].copy()

        # Step 5: Multi-scale Gamma Enhancement (still in V space)
        v_gamma1 = np.power(v_clahe_norm, gamma_values[0])
        v_gamma5 = np.power(v_clahe_norm, gamma_values[1])

        v_gamma1_disp = (v_gamma1 * 255).astype("uint8")
        v_gamma5_disp = (v_gamma5 * 255).astype("uint8")

        frame_gamma1 = cv2.cvtColor(cv2.merge((h, s, v_gamma1_disp)), cv2.COLOR_HSV2BGR)
        frame_gamma5 = cv2.cvtColor(cv2.merge((h, s, v_gamma5_disp)), cv2.COLOR_HSV2BGR)

        writers["gamma1"].write(frame_gamma1)
        writers["gamma5"].write(frame_gamma5)
        if frame_count == 2:
            step_images["gamma1"] = frame_gamma1.copy()
            collage_images["gamma1"] = frame_gamma1.copy()
            step_images["gamma5"] = frame_gamma5.copy()
            collage_images["gamma5"] = frame_gamma5.copy()

        # Step 6: Morphological Top-Hat Transform (applied on gamma images)
        tophat1 = cv2.morphologyEx(v_gamma1, cv2.MORPH_TOPHAT, structuring_element)
        tophat2 = cv2.morphologyEx(v_gamma5, cv2.MORPH_TOPHAT, structuring_element)

        tophat1_norm = cv2.normalize(
            tophat1.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX
        )
        tophat2_norm = cv2.normalize(
            tophat2.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX
        )

        # Combine both top-hat maps
        weight_map_norm = 1.0 - cv2.normalize(
            0.5 * (tophat1_norm + tophat2_norm), None, 0.0, 1.0, cv2.NORM_MINMAX
        )

        writers["tophat"].write(
            cv2.cvtColor((tophat1_norm * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
        )
        if frame_count == 2:
            step_images["tophat"] = cv2.cvtColor(
                (tophat1_norm * 255).astype("uint8"), cv2.COLOR_GRAY2BGR
            )
            collage_images["tophat"] = step_images["tophat"].copy()

        # Step 7: Image Fusion (directly on V gamma images)
        if use_pca:
            stacked = np.vstack((v_gamma1.flatten(), v_gamma5.flatten())).T
            pca = PCA(n_components=2)
            pca.fit(stacked)
            weights = np.abs(pca.components_[0])
            weights = weights / np.sum(weights)

            fused = (weights[0] * v_gamma1 + weights[1] * v_gamma5) * weight_map_norm
            fusion_method = "pca_fused"
        else:
            fused = (
                manual_weight * v_gamma1 + (1 - manual_weight) * v_gamma5
            ) * weight_map_norm
            fusion_method = "manual_fused"

        fused_disp = (np.clip(fused, 0, 1) * 255).astype("uint8")
        writers[fusion_method].write(cv2.cvtColor(fused_disp, cv2.COLOR_GRAY2BGR))
        if frame_count == 2:
            collage_images[fusion_method] = cv2.cvtColor(fused_disp, cv2.COLOR_GRAY2BGR)
            step_images[fusion_method] = collage_images[fusion_method].copy()

        # Step 8: Undo initial inversion & final merge
        v_final = 1.0 - fused
        v_final_uint8 = (np.clip(v_final, 0, 1) * 255).astype("uint8")

        hsv_final = cv2.merge((h, s, v_final_uint8))
        final_output = cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
        writers["final_output"].write(final_output)
        if frame_count == 2:
            collage_images["final_output"] = final_output.copy()
            step_images["final_output"] = final_output.copy()

    cap.release()
    for w in writers.values():
        w.release()

    # Save step images
    for step_name, img in step_images.items():
        cv2.imwrite(str(output_dir / f"step_{step_name}.png"), img)

    # Create labeled collage
    collage_steps = ["original", "v_inv", "v_clahe", fusion_method, "final_output"]
    labeled_images = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    label_height = 40

    for step in collage_steps:
        if step in collage_images:
            img = collage_images[step]
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            h_img, w_img = img.shape[:2]
            labeled_img = np.zeros((h_img + label_height, w_img, 3), dtype=np.uint8)
            labeled_img[:h_img, :, :] = img
            text_size = cv2.getTextSize(step, font, font_scale, thickness)[0]
            text_x = (w_img - text_size[0]) // 2
            text_y = h_img + (label_height + text_size[1]) // 2
            cv2.putText(
                labeled_img,
                step,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )
            labeled_images.append(labeled_img)

    if labeled_images:
        collage = np.hstack(labeled_images)
        cv2.imwrite(output_dir / "processing_pipeline_collage.png", collage)

    print(f"Saved all processing step videos and images to: {output_dir}")
    print(
        f"Fusion method: {'PCA' if use_pca else 'Manual (weight=' + str(manual_weight) + ')'}"
    )


# Example usage
if __name__ == "__main__":
    input_video = r"C:\Users\BhavyaSehgal\Downloads\bhavya_phd\dataset\HumanEvaFull\S3\Image_Data\Walking_2_(C3).avi"
    output_dir_pca = r"C:\Users\BhavyaSehgal\Downloads\output_pca_new"
    output_dir_manual = r"C:\Users\BhavyaSehgal\Downloads\output_manual_new"

    # enhance_dark_video_with_labeled_collage(input_video, output_dir_pca, use_pca=True)
    enhance_dark_video_with_labeled_collage(
        input_video, output_dir_manual, use_pca=False, manual_weight=0.6
    )
