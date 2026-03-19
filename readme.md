# 🖼️ Image Context Analyzer

> **AI-powered Python tool** that reads any image and gives you a detailed, structured understanding of what's in it — using **100% free vision models** via OpenRouter.

---

## 🧠 Architecture

```
Image Input
    │
    ▼
┌─────────────────────┐
│   Image Loader       │  Load · Validate · Base64-encode · Extract metadata
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Prompt Engineering  │  Build structured prompt (quick / standard / full)
└────────┬────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         OpenRouter Free API              │
│  openrouter/free  →  auto-routes to      │
│  best available free vision model        │
│  (Mistral · Gemma · Qwen · and more)     │
└────────┬─────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Output Formatter   │  Structure · Display · Save (.json / .md / .txt)
└─────────────────────┘
```

---

## 🔍 How It Works

1. **Image Loading** — The image is loaded, validated, and metadata (width, height, format, size) is extracted using Pillow.
2. **Prompt Engineering** — A structured prompt is selected based on your chosen detail level, guiding the model to extract specific insights from the image.
3. **Free Vision Model** — The image is sent to OpenRouter's free tier, which automatically routes to the best available free vision model.
4. **Structured Output** — The model's response is formatted into readable sections covering scene, objects, text, mood, setting, and a complete narrative.

---

## 📸 Example Output

**Input:** `02_color_grid.png` — a 3×3 grid of colored squares

**Output (`--detail standard`):**
```
SCENE OVERVIEW
A digitally generated color grid displaying nine distinct colored
squares arranged in a 3x3 layout with thin dark borders.

KEY ELEMENTS
- 9 colored squares: red, green, blue, yellow, magenta, cyan,
  orange, purple, and white
- Uniform cell sizes with clean dark outlines separating each cell

COLORS & VISUAL STYLE
A vivid, high-saturation color palette covering the full visible
spectrum. Clean, graphic design aesthetic with maximum contrast.

SETTING & CONTEXT
Computer-generated color reference chart or test pattern.

NOTABLE DETAILS
No text visible. Grid is perfectly symmetrical.

OVERALL SUMMARY
A synthetic 3x3 color grid commonly used in screen calibration,
design testing, or computer vision benchmarking.
```

---

## 🚀 Features

- **3 detail levels** — `quick` (3–5 sentences), `standard` (6 sections), `full` (10 sections + narrative)
- **Auto model routing** — always uses the best available free vision model
- **3 output formats** — `.json`, `.md`, `.txt`
- **Token tracking** — shows prompt and output token counts
- **Full test suite** — 130+ unit tests, integration tests, sample image generator
- **Production-grade code** — dataclasses, enums, custom exceptions, proper logging

---

## ⚙️ Free Tier Limits

| Limit | Value |
|---|---|
| Requests per day | 50 |
| Requests per minute | 20 |
| Credit card required | ❌ No |
| Model quality | ⭐⭐⭐⭐ Excellent |

---

## 🎯 Use Cases

| Use Case | Detail Level |
|---|---|
| Accessibility — image descriptions for visually impaired | `full` |
| Content moderation — detect unsafe visuals | `standard` |
| Automated captioning — CMS, e-commerce, social media | `quick` |
| Dataset annotation — label images for ML training | `standard` |
| Document intelligence — extract text from scanned images | `full` |
| Research — analyse photos, diagrams, charts | `full` |

---

## 📦 Installation & Setup

### Prerequisites

- Python 3.9 or higher
- A free OpenRouter API key

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/image-analyzer.git
cd image-analyzer
```

### Step 2 — Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Get your free OpenRouter API key

1. Go to 👉 [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign up — takes 30 seconds, no credit card needed
3. Click **"Create Key"**
4. Copy the key — it starts with `sk-or-v1-...`

### Step 5 — Set your API key

```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY = "sk-or-v1-your_key_here"

# Mac / Linux
export OPENROUTER_API_KEY="sk-or-v1-your_key_here"
```

> **Tip:** To make the key permanent on Windows, run:
> ```powershell
> [Environment]::SetEnvironmentVariable("OPENROUTER_API_KEY", "sk-or-v1-...", "User")
> ```
> Then restart your terminal.

### Step 6 — Verify setup

```bash
python image_analyzer.py --version
```

---

## 💡 Usage

### Basic analysis

```bash
python image_analyzer.py --image photo.jpg
```

### Choose detail level

```bash
# 3-5 sentence summary
python image_analyzer.py --image photo.jpg --detail quick

# 6-section structured report (default)
python image_analyzer.py --image photo.jpg --detail standard

# 10-section deep analysis + full narrative
python image_analyzer.py --image photo.jpg --detail full
```

### Save output to a file

```bash
# Save as Markdown
python image_analyzer.py --image photo.jpg --detail full --output report.md

# Save as JSON (great for automation)
python image_analyzer.py --image photo.jpg --output result.json

# Save as plain text
python image_analyzer.py --image photo.jpg --output result.txt
```

### Pass API key directly

```bash
python image_analyzer.py --image photo.jpg --api-key sk-or-v1-your_key_here
```

### Use a specific free model

```bash
python image_analyzer.py --image photo.jpg --model mistral-small
python image_analyzer.py --image photo.jpg --model gemma-3-27b
python image_analyzer.py --image photo.jpg --model auto   # default
```

### Suppress decorative output (quiet mode)

```bash
python image_analyzer.py --image photo.jpg --quiet
```

### Full example

```bash
python image_analyzer.py \
  --image photo.jpg \
  --detail full \
  --output report.md
```

---

## 🖥️ All CLI Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--image` | `-i` | required | Path to image file |
| `--detail` | `-d` | `standard` | Analysis depth: `quick`, `standard`, `full` |
| `--output` | `-o` | None | Save to `.txt`, `.md`, or `.json` |
| `--model` | `-m` | `auto` | Free model alias |
| `--api-key` | `-k` | env var | OpenRouter API key |
| `--quiet` | `-q` | False | Suppress decorative output |
| `--version` | `-v` | — | Show version and exit |

---

## 🆓 Available Free Models

| Alias | Model ID | Notes |
|---|---|---|
| `auto` | `openrouter/free` | **Default** — always works |
| `mistral-small` | `mistralai/mistral-small-3.1-24b-instruct:free` | 24B vision |
| `gemma-3-27b` | `google/gemma-3-27b-it:free` | Google 27B |
| `gemma-3-12b` | `google/gemma-3-12b-it:free` | Google 12B |
| `qwen-vl-3b` | `qwen/qwen2.5-vl-3b-instruct:free` | Lightweight |
| `nemotron` | `nvidia/llama-3.1-nemotron-nano-8b-v1:free` | NVIDIA |

> **Note:** Free model availability changes frequently. If you get a 404 error, use `--model auto` — it always routes to a working model.
> Check current models at: [https://openrouter.ai/models?q=free&modality=image](https://openrouter.ai/models?q=free&modality=image)

---

## 🖼️ Supported Image Formats

`.jpg` · `.jpeg` · `.png` · `.gif` · `.webp` · `.bmp` · `.tiff`

---

## 📁 Output File Examples

### JSON (`result.json`)

```json
{
  "timestamp": "2025-03-19T10:30:00",
  "model": "openrouter/free",
  "detail_level": "standard",
  "elapsed_sec": 3.8,
  "prompt_tokens": 512,
  "output_tokens": 280,
  "image": {
    "file_name": "photo.jpg",
    "width": 1920,
    "height": 1080,
    "size_kb": 245.3
  },
  "analysis": "SCENE OVERVIEW\n..."
}
```

### Markdown (`report.md`)

Generates a formatted report with a metadata table, image info section, and the full analysis.

---

## 🧪 Running Tests

```bash
# Unit tests (no API key needed — everything is mocked)
pytest test_image_analyzer.py -v

# Integration tests (requires OpenRouter API key + internet)
pytest test_sample_images.py -v

# Run all tests
pytest -v

# With coverage report
pip install pytest-cov
pytest test_image_analyzer.py --cov=image_analyzer --cov-report=term-missing
```

### Generate sample test images

```bash
python generate_sample_images.py
```

This creates 8 test images in `samples/` covering: geometric shapes, color grids, text, gradients, charts, patterns, and low-light scenes.

---

## 📊 Project Structure

```
image_analyzer/
├── image_analyzer.py          ← main script
├── test_image_analyzer.py     ← unit tests (130+ cases, no API needed)
├── test_sample_images.py      ← integration tests (API + Ollama needed)
├── generate_sample_images.py  ← generates 8 test images using Pillow
├── expected_outputs.txt       ← expected analysis for each test image
├── requirements.txt           ← Pillow + openai
├── .gitignore                 ← excludes venv, .env, __pycache__
├── .env.example               ← template showing required env vars
├── README.md                  ← this file
└── samples/                   ← generated test images
    ├── 01_red_circle.png
    ├── 02_color_grid.png
    ├── 03_text_image.png
    ├── 04_gradient_sunset.png
    ├── 05_bar_chart.png
    ├── 06_checkerboard.png
    ├── 07_shapes_scene.png
    └── 08_dark_image.png
```

---

## 🔧 Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `API key not found` | Key not set | `$env:OPENROUTER_API_KEY = "sk-or-v1-..."` |
| `404 No endpoints found` | Model offline | Use `--model auto` |
| `429 Rate limit` | Too many requests | Wait 1 min (20 req/min limit) |
| `Unsupported format` | Wrong file type | Convert to `.jpg` or `.png` |
| `File not found` | Wrong path | Use full path or navigate to image folder |
| `openai not installed` | Missing package | `pip install openai` |

---

## 🔐 Security

**Never commit your API key.** The `.gitignore` excludes `.env` files. Always use environment variables:

```bash
# ✅ Correct — environment variable
$env:OPENROUTER_API_KEY = "sk-or-v1-..."

# ❌ Wrong — never hardcode in the script
api_key = "sk-or-v1-..."
```

If you accidentally push a key, go to [https://openrouter.ai/keys](https://openrouter.ai/keys) and delete it immediately.

---

