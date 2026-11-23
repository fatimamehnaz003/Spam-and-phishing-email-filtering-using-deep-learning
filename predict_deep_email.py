"""
Predict one email using the saved deep learning model.

Run example:
python predict_deep_email.py --text "Your account has been suspended. Verify your password now."
"""

import argparse
import torch

from deep_email_filter import LABEL_MAP, load_saved_model, predict_email


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="deep_email_filter.pt")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vocab, config = load_saved_model(args.model_path, device)

    prediction, probabilities, cleaned_text = predict_email(
        model=model,
        text=args.text,
        vocab=vocab,
        max_len=config["max_len"],
        device=device,
    )

    print("\nInput Email:")
    print(args.text)

    print("\nCleaned Email:")
    print(cleaned_text)

    print("\nPrediction:")
    print(LABEL_MAP[prediction])

    print("\nClass Probabilities:")
    for label_id, probability in enumerate(probabilities):
        print(f"{LABEL_MAP[label_id]}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
