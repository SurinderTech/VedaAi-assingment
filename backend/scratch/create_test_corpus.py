"""
Test Corpus Generator for VedaAI Document Intelligence.

Generates realistic image documents for testing:
1. digital_sectioned_mcq.png — Section A (MCQs) & Section B (Short Answers) with administrative headers.
2. admin_heavy_paper.png — Header-heavy page with instructions, metadata, and 2 real questions.
"""
import os
from PIL import Image, ImageDraw, ImageFont

def create_sectioned_mcq_image(output_path: str):
    width, height = 1000, 1400
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 24)
        font_sub = ImageFont.truetype("arial.ttf", 16)
        font_text = ImageFont.truetype("arial.ttf", 18)
        font_bold = ImageFont.truetype("arialbd.ttf", 20)
    except IOError:
        font_title = font_sub = font_text = font_bold = ImageFont.load_default()

    y = 40
    # Header & Administrative Metadata
    draw.text((300, y), "NATIONAL INSTITUTE OF TECHNOLOGY", fill="black", font=font_title)
    y += 35
    draw.text((320, y), "MID-SEMESTER EXAMINATION 2026", fill="black", font=font_bold)
    y += 35
    draw.text((50, y), "Course: CS401 Machine Learning", fill="black", font=font_sub)
    draw.text((650, y), "Max Marks: 50", fill="black", font=font_sub)
    y += 25
    draw.text((50, y), "Time Allowed: 2 Hours", fill="black", font=font_sub)
    draw.text((650, y), "Roll No: _____________", fill="black", font=font_sub)
    y += 35
    draw.line([(40, y), (960, y)], fill="black", width=2)
    y += 20

    # Instructions
    draw.text((50, y), "General Instructions:", fill="black", font=font_bold)
    y += 25
    draw.text((50, y), "1. Answer all questions from Section A and Section B.", fill="black", font=font_sub)
    y += 22
    draw.text((50, y), "2. All answers must be written clearly in the provided booklet.", fill="black", font=font_sub)
    y += 35

    # Section A
    draw.line([(40, y), (960, y)], fill="black", width=1)
    y += 15
    draw.text((400, y), "SECTION A — MULTIPLE CHOICE", fill="black", font=font_bold)
    y += 40

    # Q1
    draw.text((50, y), "1. Which loss function is commonly used for binary classification?", fill="black", font=font_text)
    y += 30
    draw.text((80, y), "(A) Mean Squared Error", fill="black", font=font_text)
    draw.text((480, y), "(B) Binary Cross-Entropy", fill="black", font=font_text)
    y += 28
    draw.text((80, y), "(C) Mean Absolute Error", fill="black", font=font_text)
    draw.text((480, y), "(D) Hinge Loss", fill="black", font=font_text)
    y += 45

    # Q2
    draw.text((50, y), "2. What is the main purpose of regularization in neural networks?", fill="black", font=font_text)
    y += 30
    draw.text((80, y), "(A) Increase training speed", fill="black", font=font_text)
    draw.text((480, y), "(B) Prevent overfitting", fill="black", font=font_text)
    y += 28
    draw.text((80, y), "(C) Reduce model capacity to zero", fill="black", font=font_text)
    draw.text((480, y), "(D) Normalize input data", fill="black", font=font_text)
    y += 45

    # Section B
    draw.line([(40, y), (960, y)], fill="black", width=1)
    y += 15
    draw.text((420, y), "SECTION B — SHORT ANSWER", fill="black", font=font_bold)
    y += 40

    # Q3
    draw.text((50, y), "3. Explain the gradient descent optimization algorithm in detail.", fill="black", font=font_text)
    y += 50

    # Q4
    draw.text((50, y), "4. Define learning rate hyperparameter and discuss its impact on convergence.", fill="black", font=font_text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Created test corpus image: {output_path}")

def create_admin_heavy_image(output_path: str):
    width, height = 1000, 1400
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_sub = ImageFont.truetype("arial.ttf", 16)
        font_text = ImageFont.truetype("arial.ttf", 18)
        font_bold = ImageFont.truetype("arialbd.ttf", 18)
    except IOError:
        font_title = font_sub = font_text = font_bold = ImageFont.load_default()

    y = 40
    draw.text((250, y), "DELHI TECHNOLOGICAL UNIVERSITY", fill="black", font=font_title)
    y += 30
    draw.text((320, y), "DEPARTMENT OF COMPUTER SCIENCE", fill="black", font=font_bold)
    y += 30
    draw.text((50, y), "Candidate Name: _____________________", fill="black", font=font_sub)
    draw.text((600, y), "Roll No: __________________", fill="black", font=font_sub)
    y += 25
    draw.text((50, y), "Course: B.Tech CSE Semester VI", fill="black", font=font_sub)
    draw.text((600, y), "Subject Code: CO302", fill="black", font=font_sub)
    y += 25
    draw.text((50, y), "Time: 3 Hours", fill="black", font=font_sub)
    draw.text((600, y), "Max Marks: 100", fill="black", font=font_sub)
    y += 30
    draw.line([(40, y), (960, y)], fill="black", width=2)
    y += 20

    # Administrative list that should NOT become questions
    draw.text((50, y), "General Guidelines for Assessment:", fill="black", font=font_bold)
    y += 30
    draw.text((50, y), "1. Written Tests (Weightage 20%)", fill="black", font=font_text)
    y += 25
    draw.text((50, y), "2. Assignments and Project Work (Weightage 15%)", fill="black", font=font_text)
    y += 25
    draw.text((50, y), "3. End Semester Examination (Weightage 65%)", fill="black", font=font_text)
    y += 40
    draw.line([(40, y), (960, y)], fill="black", width=1)
    y += 25

    # Actual Examination Questions
    draw.text((50, y), "Q1. Define overfitting and underfitting in supervised learning models.", fill="black", font=font_text)
    y += 50
    draw.text((50, y), "Q2. Differentiate between L1 (Lasso) and L2 (Ridge) regularization.", fill="black", font=font_text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Created admin heavy image: {output_path}")

def create_multipage_pdf(output_path: str):
    """Creates a 2-page question paper PDF where Q6 starts on Page 1 and continues on Page 2."""
    width, height = 1000, 1400
    page1 = Image.new("RGB", (width, height), color="white")
    draw1 = ImageDraw.Draw(page1)

    page2 = Image.new("RGB", (width, height), color="white")
    draw2 = ImageDraw.Draw(page2)

    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_text = ImageFont.truetype("arial.ttf", 18)
        font_bold = ImageFont.truetype("arialbd.ttf", 18)
    except IOError:
        font_title = font_text = font_bold = ImageFont.load_default()

    # Page 1
    y = 50
    draw1.text((300, y), "UNIVERSITY SEMESTER EXAMINATION", fill="black", font=font_title)
    y += 40
    draw1.text((50, y), "Subject: Artificial Intelligence & Robotics", fill="black", font=font_bold)
    y += 40
    draw1.text((50, y), "Q5. Describe the A* search algorithm and explain its heuristic admissibility condition.", fill="black", font=font_text)
    y += 60
    draw1.text((50, y), "Q6. Formulate the Minimax algorithm for two-player zero-sum games with Alpha-Beta pruning.", fill="black", font=font_text)
    y += 35
    draw1.text((50, y), "Explain how Alpha and Beta bounds are maintained during depth-first search traversal and provide", fill="black", font=font_text)

    # Page 2
    y = 50
    draw2.text((50, y), "a complete pseudocode implementation showing cut-off conditions and move evaluation.", fill="black", font=font_text)
    y += 60
    draw2.text((50, y), "Q7. What is the Markov Decision Process (MDP) framework? Define Bellman optimality equations.", fill="black", font=font_text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    page1.save(output_path, save_all=True, append_images=[page2])
    print(f"Created multi-page PDF: {output_path}")

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "test_corpus")
    create_sectioned_mcq_image(os.path.join(out_dir, "digital_sectioned_mcq.png"))
    create_admin_heavy_image(os.path.join(out_dir, "admin_heavy.png"))
    create_multipage_pdf(os.path.join(out_dir, "multi_page_paper.pdf"))
