import zipfile
import os

zip_path = r"C:\Users\Usuario\OneDrive\Escritorio\machine\emos.zip"
extract_to = r"C:\Users\Usuario\OneDrive\Escritorio\machine\Emos"

os.makedirs(extract_to, exist_ok=True)

print("📦 Extrayendo RAF-DB...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Extracción completada en:", extract_to)