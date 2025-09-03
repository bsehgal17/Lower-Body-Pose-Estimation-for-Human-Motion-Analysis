from .types import apply_motion_blur as _apply_motion_blur


def apply_motion_blur(frame, noise_params):
    if getattr(noise_params, "apply_motion_blur"):
        return _apply_motion_blur(
            frame, kernel_size=noise_params.motion_blur_kernel_size
        )
    return frame
