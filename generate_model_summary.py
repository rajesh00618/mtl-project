import pathlib
from io import StringIO

from mtl.model import MultiTaskModel

RESULTS_DIR = pathlib.Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_summary():
    model = MultiTaskModel()
    model.build((None, 28, 28, 1))

    buffer = StringIO()
    model.backbone.summary(print_fn=lambda line: buffer.write(line + "\n"))
    buffer.write("\n")
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))

    with open(RESULTS_DIR / "model_architecture.txt", "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())
    print("Saved model summary to results/model_architecture.txt")


if __name__ == "__main__":
    save_summary()
