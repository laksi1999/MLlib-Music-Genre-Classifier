import os

from my_pipeline.lr_pipeline import LRPipeline


if __name__ == "__main__":
    DATASET = "./Merged_dataset.csv"
    MODEL_DIR = "./model/"

    os.makedirs(MODEL_DIR, exist_ok=True)

    p = LRPipeline()

    m = p.train_and_test(DATASET, 0.8, MODEL_DIR)

    ul = "unknown lyrics"
    r, h = p.predict_for_unknown_lyrics(ul, m)

    print(f"unknown_lyrics: {ul}")
    print(f"prediction: {r}")
    print(f"probabilities: {h}")

    p.stop_pipeline()
