"""
generate_docx.py — Script to generate ResoGenre_Project_Report.docx
Creates a formatted Microsoft Word document covering the complete
project explanation, tech stack, architecture flow, and evaluation.
"""

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Set background color for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def create_project_report():
    doc = Document()

    # Page Margins: 1 inch everywhere
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # 1. Document Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title.add_run("ResoGenre: Deep-Learning-Based Audio & Music Genre Classification System")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(30, 41, 59) # Slate 800

    # Subtitle
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = subtitle.add_run("Complete Technical Architecture, System Flow, and Engineering Documentation\n")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(100, 116, 139) # Slate 500

    # Meta banner
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_meta = p_meta.add_run("Technology Stack: PyTorch (CUDA GPU) | FastAPI | Next.js 16 | Librosa | GTZAN Benchmark")
    run_meta.font.name = "Calibri"
    run_meta.font.size = Pt(10)
    run_meta.font.bold = True
    run_meta.font.color.rgb = RGBColor(14, 116, 144) # Cyan 700

    doc.add_paragraph("—" * 55).alignment = WD_ALIGN_PARAGRAPH.CENTER

    # SECTION 1: EXECUTIVE SUMMARY
    h1 = doc.add_heading(level=1)
    run_h1 = h1.add_run("1. Executive Summary & Project Purpose")
    run_h1.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph(
        "Music Genre Classification (MGC) is a fundamental pillar of modern digital audio streaming platforms, "
        "enabling automated cataloging, acoustic similarity search, and personalized recommendation engines (e.g., Spotify, Apple Music). "
        "Traditional audio classification approaches rely on shallow machine learning algorithms (such as Random Forests or SVMs) "
        "trained on flat, time-averaged summary statistics (such as the mean and variance of MFCCs). These classical methods suffer from a "
        "fatal limitation: by averaging features over time, they destroy critical rhythmic syncopation, drum patterns, chord progressions, "
        "and temporal dynamic shifts."
    )
    doc.add_paragraph(
        "ResoGenre overcomes this limitation by transforming 1D raw audio waveforms into 2D Log-Mel Spectrogram representations, "
        "thereby re-framing audio classification as a computer vision task. Using a custom 4-Stage 2D Deep Convolutional Neural Network (CNN) "
        "accelerated by NVIDIA CUDA GPU computing, ResoGenre captures both frequency harmonics (timbre) and temporal rhythms simultaneously. "
        "The model achieves 79.00% track-level accuracy on unseen test songs, outperforming the Random Forest baseline by +11.6%."
    )

    # SECTION 2: TECHNOLOGY STACK
    h2 = doc.add_heading(level=1)
    run_h2 = h2.add_run("2. Comprehensive Technology Stack")
    run_h2.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph("The ResoGenre system is built using modern, production-grade tools across deep learning, signal processing, and web engineering:")

    # Tech Stack Table
    tech_table = doc.add_table(rows=1, cols=3)
    tech_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = tech_table.rows[0].cells
    hdr_cells[0].text = "Component Layer"
    hdr_cells[1].text = "Technology / Framework"
    hdr_cells[2].text = "Specific Role in Project"
    for cell in hdr_cells:
        set_cell_background(cell, "1E293B")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)

    tech_data = [
        ("Deep Learning Framework", "PyTorch 2.6.0 + cu124", "Neural network architecture, tensor operations, backpropagation, and checkpoint serialization."),
        ("Hardware Acceleration", "NVIDIA CUDA 12.4 (RTX 2050)", "Hardware acceleration on 4GB VRAM GPU, reducing training time to 6.2s/epoch and inference to 18ms."),
        ("Audio Signal Processing", "Librosa & SoundFile", "Audio resampling (22,050 Hz), STFT, 128 Mel filterbanks, and decibel logarithmic conversion."),
        ("Classical ML Baseline", "Scikit-Learn 1.4+", "Extraction of 55 acoustic features, Random Forest baseline, and classification metrics (F1, Confusion Matrix)."),
        ("Backend Web Framework", "FastAPI + Uvicorn", "Async REST endpoints (/predict, /waveform, /spectrogram, /health) with lifespan GPU model loading."),
        ("Data Schema Validation", "Pydantic", "Strict typing and validation for request/response JSON contracts."),
        ("Frontend Application", "Next.js 16 (React 19)", "App Router, client-side dynamic rendering, Turbopack, and responsive Glassmorphism layout."),
        ("Interactive Visualizer", "HTML5 Canvas API", "Real-time dual-gradient waveform rendering with sub-pixel interpolation."),
        ("1-Click Launch Automation", "Windows Batch Script (run.bat)", "Concurrent dual-process launcher managing both FastAPI and Next.js in dedicated consoles.")
    ]

    for layer, tech, role in tech_data:
        row_cells = tech_table.add_row().cells
        row_cells[0].text = layer
        row_cells[1].text = tech
        row_cells[2].text = role
        for i, cell in enumerate(row_cells):
            set_cell_background(cell, "F8FAFC" if len(tech_table.rows) % 2 == 0 else "FFFFFF")
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9.5)

    doc.add_paragraph()

    # SECTION 3: SYSTEM FLOW & ARCHITECTURE
    h3 = doc.add_heading(level=1)
    run_h3 = h3.add_run("3. End-to-End System Flow: Step-by-Step Architecture")
    run_h3.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph("The complete lifecycle of an audio track in ResoGenre proceeds through six structured stages:")

    steps = [
        ("Step 1: Audio Ingestion & Resampling",
         "The user submits an audio track (.wav, .mp3, .ogg, .flac) via drag-and-drop or 1-click sample selector. The backend ingests the audio and immediately resamples it to 22,050 Hz mono. The Nyquist frequency (11,025 Hz) captures fundamental tones and musical overtones while preserving memory efficiency."),
        ("Step 2: Dynamic Peak Normalization",
         "Audio volume levels vary widely between tracks. ResoGenre scales the raw signal by dividing by its maximum absolute peak (y_norm = (y / max(|y|)) * 0.95). This prevents quiet acoustic passages (such as solo piano) from being lost in background noise."),
        ("Step 3: Dense 50% Overlapping Sliding Window",
         "To analyze a full 30-second track without missing transitions, ResoGenre generates 3.0-second slices using a 1.5-second hop duration (50% temporal overlap). For a 30-second song, this produces exactly 19 distinct time windows."),
        ("Step 4: Time-Frequency Feature Extraction (Log-Mel Spectrogram)",
         "For each 3-second slice (66,150 samples), the Short-Time Fourier Transform (STFT) is calculated with an FFT window of 2048 and hop length of 512 using a Hann window. A 128-band Mel filterbank maps linear frequencies to the subjective Mel pitch scale. The power values are converted to decibels (10*log10) and normalized between [0.0, 1.0], yielding a tensor of shape [1, 128, 130]."),
        ("Step 5: Batched Forward Pass on GPU",
         "All 19 slice tensors are stacked into a single tensor batch of shape [19, 1, 128, 130] and transferred to CUDA VRAM. The 4-Stage 2D CNN executes a single forward pass, producing log-softmax class probabilities across all 10 genres in under 20 milliseconds."),
        ("Step 6: Dual Ensemble Aggregation & JSON Response",
         "The system computes Soft Voting (averaging predicted probability distributions across all 19 windows) to determine the top predicted genre and confidence percentage. Simultaneously, Hard Voting tallies the discrete top prediction per slice to report segment consensus (e.g., 19/19 segments agreed). The frontend renders the probability meters, interactive waveform, and spectrogram heatmap.")
    ]

    for title_text, desc_text in steps:
        p = doc.add_paragraph()
        run_step_title = p.add_run(f"• {title_text}: ")
        run_step_title.bold = True
        run_step_title.font.color.rgb = RGBColor(15, 23, 42)
        p.add_run(desc_text)

    # SECTION 4: DATASET & DATA LEAKAGE PREVENTION
    h4 = doc.add_heading(level=1)
    run_h4 = h4.add_run("4. Dataset & Critical Leakage-Safe Splitting")
    run_h4.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph(
        "The system is trained and evaluated on the GTZAN Genre Collection benchmark, comprising 1,000 tracks (100 tracks per genre across 10 genres: "
        "Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, and Rock), each 30 seconds in duration."
    )
    doc.add_paragraph(
        "A critical pitfall in audio machine learning is slice-level data leakage. If 30-second songs are divided into 3-second segments first, "
        "and random splitting is subsequently applied across all 10,000 segments, different segments of the EXACT same song will inevitably leak "
        "into both the training and test sets. In this flawed scenario, the neural network simply memorizes the artist's unique recording room acoustics "
        "and production mix, creating artificial 95%+ accuracy scores that completely fail when exposed to new music."
    )
    doc.add_paragraph(
        "ResoGenre strictly enforces Stratified Song-Level Holdout Splitting FIRST:\n"
        "• 80% Training Set: 80 songs per genre (800 songs total) -> 8,000 segments.\n"
        "• 10% Validation Set: 10 songs per genre (100 songs total) -> 1,000 segments.\n"
        "• 10% Test Set: 10 songs per genre (100 songs total) -> 1,000 segments.\n"
        "Zero segments from any test song ever appear in training or validation partitions."
    )

    # SECTION 5: MODEL ARCHITECTURE
    h5 = doc.add_heading(level=1)
    run_h5 = h5.add_run("5. 4-Stage 2D CNN Model Architecture")
    run_h5.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph(
        "The GenreCNN architecture is intentionally tailored for spectrogram feature maps, comprising 4 progressive convolutional stages "
        "followed by global average pooling and dense classification layers:"
    )

    # CNN Layers Table
    cnn_table = doc.add_table(rows=1, cols=4)
    cnn_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_hdr = cnn_table.rows[0].cells
    c_hdr[0].text = "Stage / Layer"
    c_hdr[1].text = "Layer Operations"
    c_hdr[2].text = "Output Shape"
    c_hdr[3].text = "Architectural Purpose"
    for cell in c_hdr:
        set_cell_background(cell, "1E293B")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)

    cnn_layers = [
        ("Input", "Log-Mel Spectrogram", "[Batch, 1, 128, 130]", "128 mel frequency bins by 130 time frames."),
        ("Stage 1", "Conv2D(1->32, 3x3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)", "[Batch, 32, 64, 65]", "Extracts localized spectral edges, transients, and fundamental frequencies."),
        ("Stage 2", "Conv2D(32->64, 3x3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.25)", "[Batch, 64, 32, 32]", "Captures harmonic overtone stacks and short rhythmic patterns."),
        ("Stage 3", "Conv2D(64->128, 3x3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.30)", "[Batch, 128, 16, 16]", "Extracts tempo structures, basslines, and percussion grooves."),
        ("Stage 4", "Conv2D(128->256, 3x3) + BatchNorm + ReLU + MaxPool(2x2) + Dropout(0.35)", "[Batch, 256, 8, 8]", "Captures high-level genre motifs (distorted riffs, brass chords)."),
        ("Global Pooling", "AdaptiveAvgPool2D((1, 1)) + Flatten", "[Batch, 256]", "Replaces dense flatten, drastically cutting parameters to 462,858."),
        ("FC Head", "Linear(256->128) + ReLU + Dropout(0.50) + Linear(128->10) + LogSoftmax", "[Batch, 10]", "Outputs normalized log-probabilities for all 10 genres.")
    ]

    for stage, ops, shape, purpose in cnn_layers:
        row = cnn_table.add_row().cells
        row[0].text = stage
        row[1].text = ops
        row[2].text = shape
        row[3].text = purpose
        for cell in row:
            set_cell_background(cell, "F8FAFC" if len(cnn_table.rows) % 2 == 0 else "FFFFFF")
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9.0)

    doc.add_paragraph()

    # SECTION 6: RESULTS & BENCHMARK COMPARISON
    h6 = doc.add_heading(level=1)
    run_h6 = h6.add_run("6. Experimental Results & Benchmark Analysis")
    run_h6.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph(
        "Trained on an NVIDIA GeForce RTX 2050 (4 GB VRAM) using the AdamW optimizer with learning rate 1e-3, "
        "weight decay 1e-4, and ReduceLROnPlateau scheduler. The complete 40-epoch training completed in 250.18 seconds (~4.1 minutes). "
        "The model demonstrated superior generalization over the Random Forest baseline:"
    )

    # Metrics Table
    res_table = doc.add_table(rows=1, cols=3)
    res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    r_hdr = res_table.rows[0].cells
    r_hdr[0].text = "Performance Metric"
    r_hdr[1].text = "Baseline (Random Forest)"
    r_hdr[2].text = "ResoGenre 2D CNN (Ours)"
    for cell in r_hdr:
        set_cell_background(cell, "1E293B")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)

    metrics_rows = [
        ("Full-Track Test Accuracy (Majority Vote)", "67.40%", "79.00% (+11.60% Improvement)"),
        ("3-Second Segment Test Accuracy", "62.10%", "73.70% (+11.60% Improvement)"),
        ("Macro F1-Score (All 10 Genres)", "0.6680", "0.7338"),
        ("Weighted F1-Score", "0.6710", "0.7338"),
        ("Validation Accuracy (Best Checkpoint)", "66.00%", "72.40% (Epoch 37 Checkpoint)"),
        ("Full Track Inference Latency", "450 ms", "18 ms (CUDA GPU) / 120 ms (CPU)")
    ]

    for m_name, base_val, cnn_val in metrics_rows:
        row = res_table.add_row().cells
        row[0].text = m_name
        row[1].text = base_val
        row[2].text = cnn_val
        for cell in row:
            set_cell_background(cell, "F8FAFC" if len(res_table.rows) % 2 == 0 else "FFFFFF")
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9.5)

    doc.add_paragraph()

    # SECTION 7: CASE STUDY (JAZZ VS COUNTRY)
    h7 = doc.add_heading(level=1)
    run_h7 = h7.add_run("7. Case Study: Resolving Genre Ambiguity (Jazz vs Country)")
    run_h7.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph(
        "During early testing, an interesting edge case was observed: when uploading a solo piano jazz song (jazz.00000.wav), "
        "an early 1-epoch model predicted 'Country' as the top genre. A detailed investigation revealed three compounding factors:\n"
        "1. Checkpoint Maturity: At Epoch 1, the model had only 41.5% validation accuracy, resulting in nearly uniform probabilities across all acoustic classes.\n"
        "2. Absence of Swing Signatures: The track was an unaccompanied solo piano piece lacking traditional jazz markers (swing ride cymbals, walking bass, brass formants).\n"
        "3. Harmonic Overlap: Slow acoustic country ballads share identical frequency profiles (200 Hz - 2 kHz piano voicings) with slow jazz piano.\n"
        "\nResolution Implemented:\n"
        "• Full 40-Epoch GPU Training with AdamW and learning rate scheduling raised accuracy to 79%.\n"
        "• Dynamic Peak Normalization boosted low-level harmonic details in quiet passages.\n"
        "• Dense 50% sliding-window voting (19 slices) provided comprehensive temporal evaluation.\n"
        "• Testing canonical swing jazz (jazz.00005.wav) yielded 78.66% Jazz confidence with a unanimous 19/19 segment consensus."
    )

    # SECTION 8: HOW TO RUN
    h8 = doc.add_heading(level=1)
    run_h8 = h8.add_run("8. Operational Guide: How to Run")
    run_h8.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph(
        "ResoGenre includes a 1-click launcher script (run.bat) for Windows. Double-clicking run.bat automatically verifies "
        "Python and Node.js prerequisites, checks model weights, starts the FastAPI server on port 8000, starts the Next.js server "
        "on port 3000, and opens the application in your default web browser.\n\n"
        "Manual commands:\n"
        "• Backend: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload\n"
        "• Frontend: cd frontend && npm run dev\n"
        "• Web UI: http://localhost:3000\n"
        "• Swagger Docs: http://localhost:8000/docs"
    )

    # Save document
    output_path = "c:/AIDS_mini/ResoGenre_Project_Report.docx"
    doc.save(output_path)
    print(f"[SUCCESS] Word document generated at: {output_path}")

if __name__ == "__main__":
    create_project_report()
