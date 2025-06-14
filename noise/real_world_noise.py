from .types import add_realistic_noise


def apply_combined_noise(frame, noise_params):
    apply_poisson = getattr(noise_params, "apply_poisson_noise", True)
    apply_gaussian = getattr(noise_params, "apply_gaussian_noise", True)

    if apply_poisson or apply_gaussian:
        frame = add_realistic_noise(
            frame,
            poisson_scale=noise_params.poisson_scale if apply_poisson else 0,
            gaussian_std=noise_params.gaussian_std if apply_gaussian else 0,
        )
    return frame
