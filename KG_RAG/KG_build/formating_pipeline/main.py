from datasets import load_dataset

from conversion_pipeline import (
    ConversionPipeline
)

from config import OUTPUT_DIR


def main():

    dfs = {
        cfg: load_dataset(
            "HealthDataHub/PARHAF-infectiology-annotated",
            cfg,
            split="train"
        )
        .to_pandas()
        for cfg in [    
            "document_metadata",
            "spans",
            "relations"
            ]
        }
    relations = dfs["relations"]

    pipeline = ConversionPipeline(
        dataframe=relations,
        output_dir=OUTPUT_DIR
    )

    pipeline.run()

    print("Conversion completed.")


if __name__ == "__main__":
    main()