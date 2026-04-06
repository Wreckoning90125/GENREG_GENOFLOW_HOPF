# ================================================================
# MNIST Environment — For GENREG Evolutionary Training
#
# Presents MNIST digits one at a time.
# The agent's "action" is its classification (0-9).
# Trust comes from correct/incorrect predictions.
#
# Downloads MNIST on first use, caches locally.
# Pure Python + struct (no torch/numpy dependency).
# ================================================================

import os
import gzip
import struct
import random
import math


# ================================================================
# MNIST Data Loading (pure Python)
# ================================================================

MNIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "mnist")


def _download_mnist():
    """Download MNIST dataset if not already cached."""
    import urllib.request

    os.makedirs(MNIST_DIR, exist_ok=True)

    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for fname in files:
        fpath = os.path.join(MNIST_DIR, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, fpath)
            print(f"  Saved to {fpath}")


def _read_images(path):
    """Read IDX image file, return list of flat float lists (0.0-1.0)."""
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        data = f.read()

    images = []
    pixels_per = rows * cols
    for i in range(num):
        offset = i * pixels_per
        img = [data[offset + j] / 255.0 for j in range(pixels_per)]
        images.append(img)
    return images, rows, cols


def _read_labels(path):
    """Read IDX label file, return list of ints."""
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049
        data = f.read()
    return [data[i] for i in range(num)]


def _rotate_image(pixels, rows, cols, angle_deg):
    """Rotate a flat image by angle_deg degrees (bilinear interpolation)."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cx, cy = cols / 2.0, rows / 2.0

    rotated = [0.0] * (rows * cols)
    for y in range(rows):
        for x in range(cols):
            # Map destination to source
            dx, dy = x - cx, y - cy
            sx = cos_a * dx + sin_a * dy + cx
            sy = -sin_a * dx + cos_a * dy + cy

            # Bilinear interpolation
            x0, y0 = int(math.floor(sx)), int(math.floor(sy))
            x1, y1 = x0 + 1, y0 + 1
            fx, fy = sx - x0, sy - y0

            def get_pixel(px, py):
                if 0 <= px < cols and 0 <= py < rows:
                    return pixels[py * cols + px]
                return 0.0

            val = (
                get_pixel(x0, y0) * (1 - fx) * (1 - fy) +
                get_pixel(x1, y0) * fx * (1 - fy) +
                get_pixel(x0, y1) * (1 - fx) * fy +
                get_pixel(x1, y1) * fx * fy
            )
            rotated[y * cols + x] = val
    return rotated


# Global cache
_MNIST_CACHE = {}


def load_mnist(split="train"):
    """Load MNIST split. Returns (images, labels)."""
    if split in _MNIST_CACHE:
        return _MNIST_CACHE[split]

    _download_mnist()

    if split == "train":
        img_path = os.path.join(MNIST_DIR, "train-images-idx3-ubyte.gz")
        lbl_path = os.path.join(MNIST_DIR, "train-labels-idx1-ubyte.gz")
    else:
        img_path = os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte.gz")
        lbl_path = os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte.gz")

    images, rows, cols = _read_images(img_path)
    labels = _read_labels(lbl_path)

    _MNIST_CACHE[split] = (images, labels, rows, cols)
    return images, labels, rows, cols


# ================================================================
# MNIST Environment (matches Snake interface)
# ================================================================

class MNISTEnvironment:
    """
    MNIST classification environment for GENREG.

    Each episode = one pass through a batch of digits.
    The agent "acts" by choosing a digit class (0-9).
    Trust is determined by correctness (via protein layer or direct).

    Interface matches SnakeEnvironment:
        reset() → signals dict
        step(action) → (signals dict, done bool)
        get_signals() → signals dict
    """

    def __init__(self, split="train", batch_size=100, rotate=False,
                 rotation_range=(-180, 180), subset_size=None):
        """
        Args:
            split: "train" or "test"
            batch_size: digits per episode
            rotate: whether to apply random rotations
            rotation_range: (min_deg, max_deg) for rotation
            subset_size: if set, only use first N images (faster for testing)
        """
        self.split = split
        self.batch_size = batch_size
        self.rotate = rotate
        self.rotation_range = rotation_range

        images, labels, self.rows, self.cols = load_mnist(split)
        if subset_size:
            images = images[:subset_size]
            labels = labels[:subset_size]
        self.images = images
        self.labels = labels

        # Episode state
        self.current_idx = 0
        self.batch_indices = []
        self.steps_alive = 0
        self.correct = 0
        self.total = 0
        self.alive = True
        self.current_image = None
        self.current_label = -1
        self.last_correct = False
        self.food_eaten = 0  # compatibility with Snake (= correct predictions)

        self.reset()

    def reset(self):
        """Reset for new episode. Samples a random batch of digits."""
        self.batch_indices = random.sample(range(len(self.images)), min(self.batch_size, len(self.images)))
        self.current_idx = 0
        self.steps_alive = 0
        self.correct = 0
        self.total = 0
        self.alive = True
        self.last_correct = False
        self.food_eaten = 0
        self._load_current()
        return self.get_signals()

    def _load_current(self):
        """Load the current digit image and label."""
        if self.current_idx < len(self.batch_indices):
            idx = self.batch_indices[self.current_idx]
            self.current_image = self.images[idx][:]
            self.current_label = self.labels[idx]

            if self.rotate:
                angle = random.uniform(*self.rotation_range)
                self.current_image = _rotate_image(
                    self.current_image, self.rows, self.cols, angle
                )

    def step(self, action):
        """
        Agent classifies current digit.

        Args:
            action: int (0-9), the predicted digit class

        Returns:
            signals: dict of environment signals
            done: bool indicating episode end
        """
        if not self.alive:
            return self.get_signals(), True

        # Score the prediction
        self.last_correct = (action == self.current_label)
        if self.last_correct:
            self.correct += 1
            self.food_eaten += 1
        self.total += 1
        self.steps_alive += 1

        # Advance to next digit
        self.current_idx += 1
        if self.current_idx >= len(self.batch_indices):
            self.alive = False
            return self.get_signals(), True

        self._load_current()
        return self.get_signals(), False

    def get_signals(self):
        """
        Return environment signals.

        For MNIST, the primary signals are the pixel values.
        We also include metadata for the protein layer.
        """
        signals = {
            "steps_alive": float(self.steps_alive),
            "accuracy": float(self.correct / max(1, self.total)),
            "last_correct": 1.0 if self.last_correct else 0.0,
            "correct_count": float(self.correct),
            "total_count": float(self.total),
            "current_label": float(self.current_label),
            "alive": 1.0 if self.alive else 0.0,
        }

        # Add pixel values as signals (pixel_000 through pixel_783)
        if self.current_image:
            for i, px in enumerate(self.current_image):
                signals[f"pixel_{i:03d}"] = px

        return signals

    def get_pixel_signals(self):
        """Return just the pixel values as a flat list (for direct controller input)."""
        if self.current_image:
            return self.current_image[:]
        return [0.0] * (self.rows * self.cols)

    def get_accuracy(self):
        """Return current episode accuracy."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class MNISTEnvNode:
    """MNIST environment node for the visualizer (matches SnakeEnvNode pattern)."""

    def __init__(self, split="train", batch_size=100):
        self.env = MNISTEnvironment(split=split, batch_size=batch_size)
        self.current_signals = self.env.get_signals()
        self.episode_count = 0
        self.best_accuracy = 0.0

    def step(self, action):
        signals, done = self.env.step(action)
        self.current_signals = signals
        if done:
            self.episode_count += 1
            self.best_accuracy = max(self.best_accuracy, self.env.get_accuracy())
        return signals, done

    def reset(self):
        signals = self.env.reset()
        self.current_signals = signals
        return signals

    def get_signals(self):
        return self.current_signals

    def get_stats(self):
        return {
            "episodes": self.episode_count,
            "best_accuracy": self.best_accuracy,
            "current_accuracy": self.env.get_accuracy(),
            "alive": self.env.alive,
        }
