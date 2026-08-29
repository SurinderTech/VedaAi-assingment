import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scratch.run_comprehensive_diagnostic import run_diagnostic

qp_path = r"C:\Users\surin\AppData\Local\Temp\vedaai_uploads\5e3f68fc1759\question_paper.pdf"

if __name__ == "__main__":
    print(f"Running comprehensive diagnostic on real user question paper: {qp_path}\n")
    run_diagnostic(qp_path, force_vlm=True)
