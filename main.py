import yaml
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,  # or DEBUG if you want more verbosity
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(pipelines_yaml_path: str = "config_yamls/pipeline_filter.yaml"):
    with open(pipelines_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    pipelines = config.get("pipelines", [])
    default_global_config = config.get(
        "global_config_file", "config_yamls/global_config.yaml"
    )

    for pipeline in pipelines:
        pipeline_name = pipeline.get("name", "unnamed_pipeline")
        global_config_file = pipeline.get(
            "config_yamls/global_config.yaml", default_global_config
        )

        logger.info(f"\n--- Running Pipeline: {pipeline_name} ---")
        logger.info(f"Using Global Config: {global_config_file}")

        # Track pipeline steps to enable enhancement awareness
        pipeline_steps = [step["command"]
                          for step in pipeline.get("steps", [])]

        for i, step in enumerate(pipeline.get("steps", [])):
            command = step["command"]
            pipeline_config_file = step["config_file"]

            logger.info(
                f"Running step: {command} with pipeline config {pipeline_config_file}"
            )

            # Build command with enhancement awareness
            cmd_args = [
                "python",
                "pipeline_runner.py",
                command,
                "--pipeline_config",
                pipeline_config_file,
                "--global_config",
                global_config_file,
                "--pipeline_name",
                pipeline_name,
            ]

            # Add enhancement context flags
            if command == "enhance":
                # Pass pipeline steps information to enhancement handler
                pipeline_steps_str = ",".join(pipeline_steps)
                cmd_args.extend(["--pipeline_steps", pipeline_steps_str])
                logger.info(
                    f"Enhancement will be aware of pipeline steps: {pipeline_steps_str}"
                )

            result = subprocess.run(cmd_args, text=True)

            if result.returncode != 0:
                logger.error(f"Step '{command}' failed:\n{result.stderr}")
                break
            else:
                logger.info(result.stdout)


if __name__ == "__main__":
    run_pipeline()
