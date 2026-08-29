import json
import os
import tempfile

upload_dir = os.path.join(tempfile.gettempdir(), "vedaai_uploads")
store_file = os.path.join(upload_dir, "store_metadata.json")

with open(store_file, "r", encoding="utf-8") as f:
    data = json.load(f)

files = data.get("files", {}).get("5e3f68fc1759", {})
print("Question Paper path:", files.get("question_paper"))
print("Answer Sheet path:", files.get("answer_sheet"))

res = data.get("assessments", {}).get("5e3f68fc1759", {})
questions = res.get("questions", [])
print(f"\nTotal Questions Extracted in 5e3f68fc1759: {len(questions)}")
for idx, q in enumerate(questions, 1):
    ans = q.get("answer", {})
    print(f"[{idx:02d}] ID: {q.get('id'):<25} | Num: {q.get('number'):<8} | Status: {ans.get('status'):<15} | Text: {q.get('text')[:70]}")
