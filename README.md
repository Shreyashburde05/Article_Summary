# 🤖 AI Article Summarizer

An elegant, high-performance web application built with **Hugging Face Transformers** and **Gradio** that leverages the state-of-the-art **BART (Bidirectional and Auto-Regressive Transformers)** model to summarize long articles.

The application allows users to paste articles, fine-tune the minimum and maximum summary length using interactive sliders, and receive condensed, meaningful summaries with complete execution statistics.

---

## ✨ Features

- **Advanced AI Summarization**: Uses Facebook's `bart-large-cnn` pre-trained model for abstractive and context-aware summarization.
- **Dynamic Control Sliders**: Experiment with `min_length` and `max_length` settings to control the concise nature of the output.
- **Robust Input Validation**:
  - Validates empty input states.
  - Prevents processing of extremely short inputs (requires at least 100 characters and 20 words for context).
  - Handles boundary conditions such as `min_length` exceeding `max_length`.
- **Informative Statistics Panel**: Shows word count, character count, and the compression ratio (percentage reduction) comparing original vs. summarized text.
- **Interactive UI/UX**: Designed using Gradio Blocks featuring custom CSS layout, gradient header, and a card-based structure.
- **Built-in Examples**: Provides quick-click example text files covering AI, Mars Exploration, and Renewable Energy for instant testing.
- **Robust Error Handling**: Wrapped inside clean try-except logic to catch model inference exceptions gracefully.

---

## 🛠️ Tech Stack

- **Core Logic**: Python 3.8+
- **Model Pipeline**: Hugging Face Transformers (`transformers`)
- **Deep Learning Framework**: PyTorch (`torch`)
- **Tokenizer Backend**: SentencePiece (`sentencepiece`)
- **User Interface**: Gradio (`gradio`)

---

## 🚀 Local Installation & Execution

Follow these steps to run the application on your local machine.

### 1. Clone or Move to the Project Directory
Ensure you are in the project folder containing `app.py`:
```bash
cd Article_Summary
```

### 2. Create and Activate a Virtual Environment (Recommended)
**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Execute the python server file:
```bash
python app.py
```

### 5. Access the Interface
Open your browser and navigate to the local link output in the console:
```text
http://127.0.0.1:7860
```
*Note: On first run, the model weights (~1.6GB) will download automatically. Subsequent runs will launch instantly.*

---

## 🚀 Deployment Guide

This project is pre-configured to be deployed on **Hugging Face Spaces**:

1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Create a new Space, select **Gradio** as the SDK, and set the license type.
3. Push `app.py`, `requirements.txt`, and `README.md` to the Space repository.
4. Hugging Face Spaces will automatically build and launch the Gradio container.

---

## 📸 Screenshots
*(Add your screenshots here after launching or deploying)*

| Paste & Configuration | Generated AI Summary |
| --------------------- | -------------------- |
| ![Paste Area Placeholder](https://placehold.co/600x400?text=Paste+Article+UI) | ![Summary Area Placeholder](https://placehold.co/600x400?text=Summary+Output+UI) |

---

## 🔗 Project Links & Submission Details

- **Author**: [Your Name / Student ID]
- **GitHub Repository**: [Repository URL Placeholder]
- **Hugging Face Spaces Live Demo**: [Hugging Face Spaces URL Placeholder]
- **Viva/Submission Note**: This project is built as part of the academic curriculum for demonstrating practical understanding of Transformer-based models, Natural Language Processing, and web interface integrations.
