# CMR AI OCR

Early prototype for local OCR processing of photographed CMR transport documents.

Current stage:
- load one JPG/JPEG/PNG image,
- normalize EXIF orientation,
- run PaddleOCR locally,
- save OCR fragments to JSON,
- generate a preview image with detected text boxes.

## Run

```powershell
.\.venv-py312\Scripts\python.exe main.py "D:\path\to\cmr.jpg"