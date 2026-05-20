import pandas as pd
from datasets import load_dataset
from conversion_pipeline import ConversionPipeline


def main():

    #Loading the datasets
    dfs = {cfg: load_dataset("HealthDataHub/PARHAF-infectiology-annotated", cfg, split="train").to_pandas()
       for cfg in ["document_metadata", "spans", "relations"]}
    relations = dfs["relations"]

    pipeline = ConversionPipeline(
        df=relations,
        output_dir="data/annotations"
    )

    pipeline.run()

    print("Conversion finished.")


if __name__ == "__main__":
    main()