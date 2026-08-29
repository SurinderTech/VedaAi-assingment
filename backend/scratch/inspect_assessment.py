import json
import os
import tempfile

upload_dir = os.path.join(tempfile.gettempdir(), "vedaai_uploads")
store_file = os.path.join(upload_dir, "store_metadata.json")

print(f"Store file path: {store_file}")
if os.path.exists(store_file):
    with open(store_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        files = data.get("files", {})
        assessments = data.get("assessments", {})
        print(f"Assessments in store: {list(assessments.keys())}")
        for aid, file_info in files.items():
            print(f"\nAssessment ID: {aid}")
            print(f"  QP path: {file_info.get('question_paper')}")
            print(f"  AS path: {file_info.get('answer_sheet')}")

        for aid, res in assessments.items():
            print(f"\nAssessment {aid} Result Summary:")
            questions = res.get("questions", [])
            print(f"  Total Questions Extracted: {len(questions)}")
            for idx, q in enumerate(questions[:10], 1):
                print(f"    [{idx}] ID: {q.get('id')} | Num: {q.get('number')} | Type: {q.get('question_type')} | Text: {q.get('text')[:60]}")
else:
    print("store_metadata.json does not exist.")
