"""Download and prepare the configured Libri2Mix-5000 subset."""

from scripts.download_dataset import main


if __name__ == "__main__":
    main("download_libri2mix.yaml")
