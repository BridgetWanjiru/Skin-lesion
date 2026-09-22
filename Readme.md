# Skin Lesion Classifier — HAM10000

Project : a classifier that predicts one of 7 dermoscopic skin
lesion diagnosis categories from an uploaded image, with a Streamlit portal to try it out.

## Folder layout expected

```
project/
├── skin_lesion_classification.ipynb
├── app.py
├── requirements.txt
├── README.md
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/        <- (and part_2, if you have it)
```

If your data folder has a different name or location, edit the `DATA_DIR` line near the top of
the notebook's Setup cell,that's the only path you need to change.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Run the notebook

Open `skin_lesion_classification.ipynb` in Jupyter and run all cells top to bottom. This will:
- Load and explore the HAM10000 metadata (section A2)
- Split the data by `lesion_id` to avoid leakage (section A3)
- Train a MobileNetV2-based classifier (sections A4–A5)
- Evaluate it and show example predictions (section A6)
- Save `skin_lesion_model.keras` and `class_names.json` into the same folder (section A7)

Training time depends on your machine - expect anywhere from a few minutes (GPU) to significantly
longer (CPU only).
  *Note: Section A5 checks if a pre-trained model file already exists on your hard drive. If found, it loads instantly in  to prevent redundant training sessions. Batch size is optimized to 16 to guarantee safe, crash-free execution on CPU/limited-RAM machines.*


## 3. Run the Streamlit app

Once the notebook has produced `skin_lesion_model.keras` and `class_names.json` in the same folder
as `app.py`:

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`. Upload a dermoscopic skin lesion
image and it will show the predicted diagnosis category with a confidence score.

## Notes

- This is a coursework prototype only - not a medical device, not clinically validated.
- See notebook section A8 for a full discussion of limitations and ethical considerations.
