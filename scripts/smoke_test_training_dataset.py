"""Smoke test for DeepDCTTrainingDataset."""

from torch.utils.data import DataLoader

from deepdct.data.training_dataset import DeepDCTTrainingDataset


def main() -> None:
    dataset = DeepDCTTrainingDataset(
        data_root="data",
        sequences=("00",),
        camera="left",
        image_size=(192, 640),
        allow_zero_auxiliary=True,
        strict=True,
        return_metadata=True,
    )

    print(f"Number of transitions: {len(dataset)}")

    print("\nSingle sample")
    print("-" * 60)

    sample = dataset[0]

    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(
                f"{key:20s} "
                f"shape={tuple(value.shape)!s:18s} "
                f"dtype={value.dtype}"
            )
        else:
            print(f"{key:20s} {value}")

    print("\nDataLoader batch")
    print("-" * 60)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    batch = next(iter(loader))

    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(
                f"{key:20s} "
                f"shape={tuple(value.shape)!s:22s} "
                f"dtype={value.dtype}"
            )
        else:
            print(f"{key:20s} {value}")


if __name__ == "__main__":
    main()