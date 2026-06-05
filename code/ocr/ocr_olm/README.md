# OCR

olmocr

## usage
1. Navigate to the `ir-2025-project/` main directory:
   ```bash
   cd ocr/ocr_olm
   ```
2. Install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```  

3. Run OCR:
Bounding box is shown in "bbox_pdf"
    ```bash
    python ocr.py
    ```

4. Manual fix format error if needed
    ```bash
    python write_fix_html_quotes.py
    python postprocess_from_raw.py
    ```