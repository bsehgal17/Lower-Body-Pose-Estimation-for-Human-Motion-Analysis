import os
import pandas as pd
import pickle


class ResultAggregator:
    def __init__(self, output_path, save_as_pickle=False):
        self.output_path = output_path
        self.results = []
        self.save_as_pickle = save_as_pickle

    def add_overall_result(self, subject, action, camera, pck_value, threshold):
        self.results.append(
            {
                "Subject": subject,
                "Action": action,
                "Camera": camera,
                f"PCK@{threshold}": pck_value,
            }
        )

    def add_jointwise_result(
        self, subject, action, camera, joint_names, jointwise_pck, threshold
    ):
        self.results.append(
            {
                "Subject": subject,
                "Action": action,
                "Camera": camera,
                **{
                    f"{j}_PCK@{threshold}": jointwise_pck[:, i].mean()
                    for i, j in enumerate(joint_names)
                },
            }
        )

    def save(self):
        df_new = pd.DataFrame(self.results)
        if self.save_as_pickle:
            with open(self.output_path, "wb") as f:
                pickle.dump(df_new, f)
        else:
            if os.path.exists(self.output_path):
                df_existing = pd.read_excel(self.output_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_excel(self.output_path, index=False)
            else:
                df_new.to_excel(self.output_path, index=False)
        print(f"Saved results to {self.output_path}")
