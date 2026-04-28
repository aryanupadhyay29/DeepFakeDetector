import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def detect_and_crop_face(img_cv, detector):
    results = detector.detect_faces(img_cv)

    if results:
        x, y, width, height = results[0]["box"]
        x = max(0, x)
        y = max(0, y)
        x_end = min(img_cv.shape[1], x + max(0, width))
        y_end = min(img_cv.shape[0], y + max(0, height))

        if x_end > x and y_end > y:
            face = img_cv[y:y_end, x:x_end]
            return cv2.resize(face, (128, 128))

    return cv2.resize(img_cv, (128, 128))


def preprocess_for_prediction(face_img):
    img_array = image.img_to_array(face_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def predict_fake_probability(model, detector, image_path):
    img_cv = cv2.imread(str(image_path))
    if img_cv is None:
        raise ValueError(f"Could not read image: {image_path}")

    if len(img_cv.shape) < 3 or img_cv.shape[2] != 3:
        raise ValueError(f"Image must be a 3-channel color image: {image_path}")

    face = detect_and_crop_face(img_cv, detector)
    processed_image = preprocess_for_prediction(face)
    prediction = model.predict(processed_image, verbose=0)
    return float(prediction[0][0])


def collect_images(directory):
    return sorted(
        path for path in directory.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_dataset(dataset_dir):
    real_dir = dataset_dir / "real"
    fake_dir = dataset_dir / "fake"

    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(
            "Dataset folder must contain 'real/' and 'fake/' subfolders."
        )

    real_images = collect_images(real_dir)
    fake_images = collect_images(fake_dir)

    if not real_images or not fake_images:
        raise ValueError("Both 'real/' and 'fake/' folders must contain images.")

    return real_images, fake_images


def plot_score_distribution(real_scores, fake_scores, output_path):
    plt.figure(figsize=(8, 6))
    bins = np.linspace(0.0, 1.0, 11)
    plt.hist(real_scores, bins=bins, alpha=0.65, label="Real images", color="tab:blue")
    plt.hist(fake_scores, bins=bins, alpha=0.65, label="Fake images", color="tab:red")
    plt.xlabel("Model confidence score for 'Fake'")
    plt.ylabel("Number of images")
    plt.title("Confidence Score Distribution for Real vs Fake Images")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate confidence score distribution for real and fake images."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to a folder containing 'real/' and 'fake/' subfolders.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("cnn_model.h5"),
        help="Path to the trained Keras model file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("confidence_distribution.png"),
        help="Path where the confidence distribution image should be saved.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    real_images, fake_images = build_dataset(args.dataset)
    model = load_model(args.model)
    detector = MTCNN()

    real_scores = []
    fake_scores = []

    for index, image_path in enumerate(real_images, start=1):
        print(f"[Real {index}/{len(real_images)}] Processing {image_path}")
        real_scores.append(predict_fake_probability(model, detector, image_path))

    for index, image_path in enumerate(fake_images, start=1):
        print(f"[Fake {index}/{len(fake_images)}] Processing {image_path}")
        fake_scores.append(predict_fake_probability(model, detector, image_path))

    plot_score_distribution(real_scores, fake_scores, args.output)
    print(f"\nSaved confidence distribution to: {args.output}")


if __name__ == "__main__":
    main()
