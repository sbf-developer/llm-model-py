# Loads a checkpoint and generates text from a prompt (CLI)

from config import GenerateConfig
from inference import ModelRunner


def main() -> None:
    gcfg = GenerateConfig()
    runner = ModelRunner()

    print(f"device: {runner.device}")
    runner.load()

    prompt = gcfg.prompt
    text = runner.complete(prompt)

    print("\n--- prompt ---")
    print(prompt)
    print("\n--- generated ---")
    print(text)


if __name__ == "__main__":
    main()
