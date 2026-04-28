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
    """Detect a face and resize the cropped region for model input."""
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
    """Convert image into the same normalized tensor used by the app."""
    img_array = image.img_to_array(face_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def predict_fake_probability(model, detector, image_path):
    """Return the model's probability that an image is fake."""
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

    paths = real_images + fake_images
    labels = np.array([0] * len(real_images) + [1] * len(fake_images), dtype=np.int32)
    return paths, labels


def compute_roc(labels, scores):
    thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    tpr_values = []
    fpr_values = []

    positive_total = np.sum(labels == 1)
    negative_total = np.sum(labels == 0)

    for threshold in thresholds:
        predictions = scores >= threshold
        true_positive = np.sum((predictions == 1) & (labels == 1))
        false_positive = np.sum((predictions == 1) & (labels == 0))

        tpr = true_positive / positive_total if positive_total else 0.0
        fpr = false_positive / negative_total if negative_total else 0.0

        tpr_values.append(tpr)
        fpr_values.append(fpr)

    fpr = np.array(fpr_values)
    tpr = np.array(tpr_values)
    order = np.argsort(fpr)
    auc = np.trapz(tpr[order], fpr[order])
    return fpr, tpr, auc


def plot_roc_curve(fpr, tpr, auc, output_path):
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Deepfake Detector")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_score_distribution(real_scores, fake_scores, output_path):
    plt.figure(figsize=(8, 6))
    bins = np.linspace(0.0, 1.0, 11)
    plt.hist(real_scores, bins=bins, alpha=0.65, label="Real images", color="tab:blue")
    plt.hist(fake_scores, bins=bins, alpha=0.65, label="Fake images", color="tab:red")
    plt.xlabel("Model confidence score for 'Fake'")
    plt.ylabel("Number of images")
    plt.title("Confidence Score Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate ROC curve for the deepfake model.")
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
        default=Path("roc_curve.png"),
        help="Path where the ROC image should be saved.",
    )
    parser.add_argument(
        "--distribution-output",
        type=Path,
        default=Path("confidence_distribution.png"),
        help="Path where the score distribution image should be saved.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    image_paths, labels = build_dataset(args.dataset)
    model = load_model(args.model)
    detector = MTCNN()

    scores = []
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{total}] Processing {image_path}")
        scores.append(predict_fake_probability(model, detector, image_path))

    scores = np.array(scores, dtype=np.float32)
    fpr, tpr, auc = compute_roc(labels, scores)
    plot_roc_curve(fpr, tpr, auc, args.output)
    plot_score_distribution(scores[labels == 0], scores[labels == 1], args.distribution_output)

    print(f"\nSaved ROC curve to: {args.output}")
    print(f"Saved confidence distribution to: {args.distribution_output}")
    print(f"AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
