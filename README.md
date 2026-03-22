# Attention App

Simple Streamlit app for:
- Binary sentiment prediction (positive/negative)
- Multi-label emotion prediction

## Quick Start

1. Clone the project

```bash
git clone <...>
cd Attention
```

2. Create and activate a virtual environment (workon)

```bash
mkvirtualenv attention
workon attention
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the app

```bash
streamlit run app.py
```

5. Open in browser

Use the local URL shown in terminal (usually http://localhost:8503).

## Notes

- First training run can be slow on CPU.
- Keep internet on if datasets need to download on first use.

## Optional: Build Docs

```bash
cd docs
make html
```