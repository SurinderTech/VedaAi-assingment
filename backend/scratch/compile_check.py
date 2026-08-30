import py_compile, sys

files = [
    r"app\services\document_understanding_service.py",
    r"app\services\intelligent_question_extraction_service.py",
    r"app\services\assessment_result_service.py",
    r"app\models\schemas.py",
]

all_ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"OK  {f}")
    except py_compile.PyCompileError as e:
        print(f"ERR {f}: {e}")
        all_ok = False

print("\nAll files compile OK" if all_ok else "\nCOMPILE ERRORS FOUND")
sys.exit(0 if all_ok else 1)
