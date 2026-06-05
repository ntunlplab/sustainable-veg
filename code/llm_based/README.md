# LLM based method

## How to run

- Paste OpenAI API key in each `.py` files
- Run the script inside this directory

## Scripts

- `exp2.py`: LLM prompting with image and text
- `exp2-1.py`: LLM prompting with only text
- `exp2-2.py`: Two stage, need to generate top 5 results using retriever, check `parse_args()` of this file
- `exp2-3.py`: LLM prompting voting ensemble
- `convert.py`: Convert `json` output to `csv` format