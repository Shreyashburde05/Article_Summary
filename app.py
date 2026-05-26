# ==============================================================================
# AI ARTICLE SUMMARIZER - GRADIO & HUGGING FACE TRANSFORMERS
# ==============================================================================
# This application uses a state-of-the-art Sequence-to-Sequence model (BART) 
# from the Hugging Face Transformers library to generate concise summaries 
# of long articles. The user interface is built using Gradio Blocks.
#
# Created as a clean, responsive, and deployment-ready web application.
# ==============================================================================

import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------------------------------------------------------
# 1. MODEL INITIALIZATION & LOADING
# ------------------------------------------------------------------------------
# We load the model globally once, so it doesn't reload on every button click.
# In transformers v5, specific pipeline tasks like 'summarization' have been removed 
# or restructured. To ensure compatibility with both v4 and v5, we load the
# AutoTokenizer and AutoModelForSeq2SeqLM components manually.

# Check if a GPU is available to accelerate model inference
device = 0 if torch.cuda.is_available() else -1
device_name = "GPU (CUDA)" if device == 0 else "CPU"
model_name = "facebook/bart-large-cnn"
print(f"[INFO] Initializing model '{model_name}' on: {device_name}...")

try:
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Initialize the model and move it to GPU if available
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device == 0:
        model = model.to("cuda")
    print("[INFO] Model and Tokenizer loaded successfully!")
except Exception as e:
    print(f"[WARNING] Failed to load model on {device_name} due to: {e}")
    print("[INFO] Attempting fallback to CPU model loading...")
    # Fallback to default CPU loading in case of device mapping issues
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("[INFO] Fallback model and tokenizer loaded successfully on CPU!")

# ------------------------------------------------------------------------------
# 2. CORE SUMMARIZATION FUNCTION
# ------------------------------------------------------------------------------
def summarize_article(article_text, max_length, min_length, progress=gr.Progress(track_tqdm=True)):
    """
    Validates input parameters and summarizes the provided article using the BART model.
    
    Parameters:
        article_text (str): The raw article content pasted by the user.
        max_length (int): Upper bound on the summary length (in words/tokens).
        min_length (int): Lower bound on the summary length (in words/tokens).
        progress (gr.Progress): Gradio's progress reporting object for UI feedback.
        
    Returns:
        tuple: (summary_result_text, statistics_markdown)
    """
    
    # --- Input Validation 1: Check for empty input ---
    if not article_text or not article_text.strip():
        return (
            "⚠️ Error: The article content is empty. Please paste your text in the input box.",
            "❌ **Status:** Failed (Empty input)"
        )
    
    cleaned_text = article_text.strip()
    
    # Calculate word and character count of the input article
    words = cleaned_text.split()
    word_count = len(words)
    char_count = len(cleaned_text)
    
    # --- Input Validation 2: Check if text is too short ---
    # Summarization models require sufficient context. 100 characters is a reasonable baseline.
    if char_count < 100 or word_count < 20:
        return (
            f"⚠️ Error: The article is too short ({char_count} characters, {word_count} words).\n"
            f"Please enter a longer article (at least 100 characters and 20 words) for high-quality summaries.",
            "❌ **Status:** Failed (Input too short)"
        )
    
    # --- Input Validation 3: Check min/max slider consistency ---
    if min_length > max_length:
        return (
            f"⚠️ Error: Minimum summary length ({min_length}) cannot be greater than "
            f"Maximum summary length ({max_length}). Please adjust the sliders.",
            "❌ **Status:** Failed (Invalid length constraints)"
        )
    
    # --- Model Parameter Adjustment ---
    # BART's token limit is 1024. If the input is extremely long, we leverage automatic 
    # truncation. If the input is relatively short (less than the requested max_length), 
    # we dynamically adjust the max_length parameter to prevent the model from hallucinating 
    # or producing redundant filler sentences.
    adjusted_max = max_length
    adjusted_min = min_length
    
    if word_count < max_length:
        # Cap max_length at 80% of the input word count
        adjusted_max = max(15, int(word_count * 0.8))
        # Ensure min_length is below max_length
        adjusted_min = max(5, int(adjusted_max * 0.4))
    
    progress(0.1, desc="Preparing article...")
    
    try:
        progress(0.3, desc="Tokenizing article content...")
        
        # Tokenize the input text. truncation=True maps to the model's max positional embeddings (1024)
        inputs = tokenizer(
            cleaned_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )
        
        # Move tokenized input tensors to GPU if active
        if device == 0:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        progress(0.6, desc="Generating summary with BART (this may take a few seconds)...")
        
        # Run auto-regressive token generation
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=4,
                max_length=adjusted_max,
                min_length=adjusted_min,
                early_stopping=True
            )
            
        progress(0.8, desc="Decoding generated tokens back to text...")
        
        # Decode token IDs back to a readable string, skipping special structural tokens
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        
        # Calculate summary metrics
        summary_words = len(summary_text.split())
        summary_chars = len(summary_text)
        
        # Calculate compression percentage
        reduction = max(0, int((1 - (summary_words / word_count)) * 100))
        
        progress(1.0, desc="Done!")
        
        # Format metrics and statistics for output display
        stats_markdown = (
            f"📊 **Statistics:**\n"
            f"- **Original:** {word_count} words ({char_count} chars)\n"
            f"- **AI Summary:** {summary_words} words ({summary_chars} chars)\n"
            f"- **Compression Ratio:** **{reduction}% reduction**"
        )
        
        return summary_text, stats_markdown
        
    except Exception as e:
        # Prevent app crashes by handling exceptions gracefully and returning clear messages
        print(f"[ERROR] Exception during model inference: {e}")
        return (
            f"⚠️ An unexpected error occurred during summarization:\n{str(e)}\n\n"
            f"Suggestions:\n1. Try reducing the input text size.\n2. Verify all numbers are set correctly.",
            "❌ **Status:** Failed (Inference error)"
        )

# ------------------------------------------------------------------------------
# 3. GRADIO APP DESIGN & INTERFACE
# ------------------------------------------------------------------------------

# Custom CSS for standard modern design with custom gradient headers, 
# layout margins, and clean buttons
custom_css = """
.container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 20px;
}
.header-card {
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%);
    color: white;
    padding: 35px 25px;
    border-radius: 16px;
    margin-bottom: 30px;
    box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    text-align: center;
}
.header-card h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin-bottom: 10px !important;
}
.header-card p {
    font-size: 1.15rem !important;
    color: rgba(255, 255, 255, 0.9) !important;
    margin: 0 !important;
}
.section-panel {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 22px;
    background-color: #ffffff;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    height: 100%;
}
.dark .section-panel {
    background-color: #1f2937;
    border-color: #374151;
}
.btn-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.2) !important;
}
.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 15px rgba(79, 70, 229, 0.35) !important;
}
.btn-secondary {
    background-color: #f3f4f6 !important;
    color: #374151 !important;
    font-weight: 600 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.dark .btn-secondary {
    background-color: #374151 !important;
    color: #f3f4f6 !important;
    border-color: #4b5563 !important;
}
.btn-secondary:hover {
    background-color: #e5e7eb !important;
}
.dark .btn-secondary:hover {
    background-color: #4b5563 !important;
}
.stats-box {
    margin-top: 15px;
    padding: 15px;
    border-radius: 10px;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
}
.dark .stats-box {
    background-color: #1e293b;
    border-color: #334155;
}
"""

# Example articles to help users test the app instantly
example_articles = [
    [
        "Artificial Intelligence (AI) is rapidly transforming various sectors of the global economy, from healthcare to finance and education. In healthcare, AI algorithms are helping doctors diagnose diseases with greater accuracy, analyze medical images, and even predict patient outcomes. In finance, machine learning models are used to detect fraudulent transactions and optimize investment portfolios. However, the rise of AI also brings significant ethical and societal challenges. Concerns about automation replacing human jobs, algorithmic bias in decision-making, and the lack of transparency in complex models (often called 'black boxes') are widespread. As a result, researchers and policymakers are calling for robust regulations and ethical guidelines to ensure that AI systems are developed and deployed responsibly, benefiting humanity as a whole.",
        100,
        30
    ],
    [
        "NASA's Mars exploration program has entered an exciting new phase with the Perseverance rover actively searching for signs of ancient microbial life on the Red Planet. Having landed in the Jezero Crater—a site believed to have once been flooded with water—Perseverance has been collecting rock and soil samples that will eventually be returned to Earth by a future joint mission with the European Space Agency (ESA). The rover is equipped with advanced scientific instruments, including cameras, spectrometers, and a ground-penetrating radar. It also carried Ingenuity, a technology demonstration helicopter that successfully proved powered, controlled flight is possible in Mars' thin atmosphere. The data gathered from these missions is crucial for preparing for human exploration of Mars, which space agencies hope to achieve in the late 2030s.",
        80,
        25
    ],
    [
        "Renewable energy sources, particularly solar and wind energy, are experiencing unprecedented growth worldwide as nations seek to reduce greenhouse gas emissions and combat climate change. Technological advancements and falling manufacturing costs have made solar panels and wind turbines increasingly competitive with traditional fossil fuels. Many countries are now setting ambitious targets to transition to 100% clean energy grids within the next few decades. However, integrating intermittent renewable sources into existing grids presents challenges, such as the need for massive energy storage solutions (like large-scale lithium-ion batteries) and grid modernization. Overcoming these infrastructure hurdles is vital to establishing a reliable, sustainable, and carbon-free energy future for the planet.",
        70,
        20
    ]
]

# Set up Gradio interface using Block Layout
with gr.Blocks(title="AI Article Summarizer") as demo:
    
    with gr.Column(elem_classes=["container"]):
        # Gradient Title Card
        with gr.Group(elem_classes=["header-card"]):
            gr.Markdown("# 🤖 AI Article Summarizer")
            gr.Markdown("Transform long articles into concise, structured summaries using pre-trained Facebook BART Deep Learning model.")
            
        with gr.Row(equal_height=True):
            # Left Panel: Paste Article
            with gr.Column(scale=3):
                with gr.Group(elem_classes=["section-panel"]):
                    gr.Markdown("### 📝 1. Paste Article")
                    gr.Markdown("Provide the long article text you want to condense below:")
                    
                    article_input = gr.Textbox(
                        label="Article Content",
                        placeholder="Paste your article text here (minimum 100 characters)...",
                        lines=12,
                        max_lines=20
                    )
            
            # Right Panel: Summary Settings
            with gr.Column(scale=2):
                with gr.Group(elem_classes=["section-panel"]):
                    gr.Markdown("### ⚙️ 2. Summary Settings")
                    gr.Markdown("Adjust sliders to customize summary length:")
                    
                    max_len_slider = gr.Slider(
                        label="Maximum Summary Length (words)",
                        minimum=30,
                        maximum=250,
                        value=120,
                        step=5,
                        info="Capping size of the generated summary."
                    )
                    
                    min_len_slider = gr.Slider(
                        label="Minimum Summary Length (words)",
                        minimum=10,
                        maximum=150,
                        value=30,
                        step=5,
                        info="Ensuring summary contains enough detail."
                    )
                    
                    # Action buttons grouped inside settings panel
                    with gr.Column():
                        btn_summarize = gr.Button("✨ Summarize Article", elem_classes=["btn-primary"])
                        btn_clear = gr.Button("🧹 Clear Everything", elem_classes=["btn-secondary"])

        # Bottom Panel: Generated Summary & Statistics
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["section-panel"]):
                    gr.Markdown("### 💡 3. Generated Summary")
                    
                    summary_output = gr.Textbox(
                        label="AI Summary Output",
                        placeholder="The summarized text will display here after you click 'Summarize Article'.",
                        lines=6,
                        interactive=False
                    )
                    
                    stats_output = gr.Markdown(
                        value="📊 **Statistics:** No article summarized yet.",
                        elem_classes=["stats-box"]
                    )
                    
        # Interactive Examples Component
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Quick Examples")
                gr.Markdown("Click on any preset below to instantly load and test the summarization model:")
                gr.Examples(
                    examples=example_articles,
                    inputs=[article_input, max_len_slider, min_len_slider],
                    outputs=[summary_output, stats_output],
                    fn=summarize_article,
                    cache_examples=False,
                    label="Example Articles"
                )

        # Clear Button behavior: reset inputs, outputs, sliders, and statistics
        def clear_fields():
            return "", 120, 30, "", "📊 **Statistics:** No article summarized yet."

        btn_clear.click(
            fn=clear_fields,
            inputs=[],
            outputs=[article_input, max_len_slider, min_len_slider, summary_output, stats_output]
        )

        # Summarize Button click binding with progress feedback
        btn_summarize.click(
            fn=summarize_article,
            inputs=[article_input, max_len_slider, min_len_slider],
            outputs=[summary_output, stats_output]
        )

# ------------------------------------------------------------------------------
# 4. APP LAUNCH
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Launch application locally
    # share=False by default. Setting share=True generates a public Gradio URL.
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=custom_css
    )
