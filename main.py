import yaml
import subprocess
import logging

logging.basicConfig(level=logging.INFO)


def run_pipeline(pipelines_yaml_path: str = "pipelines.yaml"):
    with open(pipelines_yaml_path, "r") as f:
        pipelines = yaml.safe_load(f).get("pipelines", [])

    for pipeline in pipelines:
        logging.info(f"\n--- Running Pipeline: {pipeline['name']} ---")
        for step in pipeline.get("steps", []):
            command = step["command"]
            config_file = step["config_file"]

            logging.info(f"Running step: {command} with config {config_file}")
            result = subprocess.run(
                ["python", "pipeline_runner.py", "--config_file",
                    step["config_file"], step["command"]],
                text=True,
            )

            if result.returncode != 0:
                logging.error(f"Step '{command}' failed:\n{result.stderr}")
                break
            else:
                logging.info(result.stdout)


if __name__ == "__main__":
    run_pipeline()
