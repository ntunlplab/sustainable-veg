# OCR

olmocr

## usage
1. Navigate to the `sustainable-veg/` main directory:
   ```bash
   cd code/ocr/ocr_olm
   ``` 

2. Run OCR:
Bounding box is shown in "bbox_pdf"
    ```bash
    python ocr.py
    ```

3. Manual fix format error if needed
    ```bash
    python write_fix_html_quotes.py
    python postprocess_from_raw.py
    ```