# SavTankVision
AI-powered photo archive classifier for tank manufacturing job folders.

> **TL;DR**: Crawls `P:\JOBS\<YEAR>\<JOB-ID>\Pictures\Shipping`, classifies photos, and **copies only full-tank images** into a central archive. Originals stay untouched. Re-run safe via manifest + content-hash dedupe. Designed for weekly automation on Windows.

---

## ✨ What it does
- **Finds photos** in nested job folders (case-insensitive `Pictures/Shipping`)
- **Classifies** images (CNN; label `1 = full_tank`, `0 = other`)
- **Copies only full-tank** images to the archive (no deletes/moves)
- **Preserves provenance** in filenames: `JOBID__ORIGNAME__HASH.ext`
- **Idempotent**: content-hash manifest avoids duplicates and rework
- **Logs everything** for audit; optional email summary + HTML “last run” report
- **Weekly scheduled** run via Windows Task Scheduler (no manual steps)

## 🧱 Stack
- **Python 3.10+**
- **PyTorch / torchvision** (ResNet18 fine-tune)
- **Pillow** (robust image loading + EXIF fix)
- **pandas** (manifest/logs)
- *(Legacy baseline: OpenCV + scikit-learn kNN — kept for the record)*

## 📈 Impact
- Processed **~15,000 photos** across **823 jobs (2019–2025)**
- Centralized, searchable “full-tank” archive
- Retrieval time dropped from **hours to minutes**
- Zero risk to originals (copy-only workflow)

---

## 📂 Paths & assumptions
