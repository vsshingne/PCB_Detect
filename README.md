# PCB Defect Detection

Local web application for inspecting printed circuit board images with a custom-trained Ultralytics YOLOv8 object detection model. It returns an annotated image, six-class counts, confidence scores, bounding boxes, and measured inference duration. There are no external inference services, database, or paid API requirements.

## Architecture and data flow

The React/Vite dashboard validates and previews a selected image, then sends it as `multipart/form-data` to FastAPI. The API validates the extension, image content, confidence threshold, and upload size before passing the decoded image to the custom YOLO model. The model is loaded once when FastAPI starts. Results and the annotated JPEG are returned as JSON; uploaded images are not saved.

```text
Browser (Vite :5173) -- image + confidence --> FastAPI (:8000) --> custom YOLOv8 weights
Browser             <-- detections + annotated image -----------
```

## Requirements

- Python 3.11 recommended (Python 3.10+ is suitable for the backend dependencies).
- Node.js 20.19+ (or 22.12+) and npm.
- The trained weights are optional for development. Predictions require the final six-class model.

## Install

### Windows PowerShell

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
cd ..\frontend
npm install
Copy-Item .env.example .env
```

### Linux / macOS

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
cd ../frontend
npm install
cp .env.example .env
```

The `.env` copies are optional with default settings. Adjust values in them as needed; restart each development server after changing environment variables.

## Add the trained model

Copy the model trained in Google Colab to:

```text
backend/models/pcb_defect_best.pt
```

The Ultralytics class metadata must match this order exactly: `Missing_hole`, `Mouse_bite`, `Open_circuit`, `Short`, `Spur`, `Spurious_copper`. The backend checks metadata at startup and reports the model as unavailable when the file is missing or class metadata differs. No pretrained model is downloaded or substituted. Model weights are excluded from Git.

## Run locally

Open two terminals from the repository root.

Backend, Windows PowerShell:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend, Linux/macOS (after activating the virtual environment):

```bash
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend, Windows PowerShell or Linux/macOS:

```bash
cd frontend
npm run dev
```

Open the Vite URL (normally <http://localhost:5173>). FastAPI health is at <http://127.0.0.1:8000/api/health>; interactive API docs are at <http://127.0.0.1:8000/docs>.

## Configuration

Backend settings are in `backend/.env.example`:

- `MODEL_PATH`: model file path; relative paths resolve from the backend directory.
- `MAX_UPLOAD_SIZE_MB`: maximum request image size (default 10 MB).
- `DEFAULT_CONFIDENCE_THRESHOLD`: default confidence threshold (default 0.25).
- `ALLOWED_ORIGINS`: comma-separated frontend origins.

Frontend settings are in `frontend/.env.example`:

- `VITE_API_BASE_URL`: FastAPI base URL (default `http://127.0.0.1:8000`).
- `VITE_MAX_UPLOAD_SIZE_MB`: client-side upload limit for usability; the backend independently enforces its own limit.

Accepted image formats are JPEG/JPG, PNG, and WEBP. Images larger than 50 megapixels are rejected to keep decoding memory bounded.

## Test and build

Backend tests do not need the trained weights or a GPU:

```powershell
cd backend
python -m pytest -q
```

```bash
cd backend
python -m pytest -q
```

Frontend production build:

```powershell
cd frontend
npm run build
```

```bash
cd frontend
npm run build
```

Real-model inference is not part of the automated test suite. Once the model has been added, restart the backend and confirm `/api/health` reports `"model_loaded": true` and `"prediction_available": true`. Then upload a representative PCB image in the dashboard (or send `file` and `confidence_threshold` as multipart form fields to `POST /api/predict`) and check that the response includes detections and a `data:image/jpeg;base64,...` annotated image. Health confirms the model loaded; successful prediction confirms it runs on the selected image.

## Troubleshooting and limitations

- **Model Missing:** place the final `.pt` file at the path above and restart FastAPI.
- **Backend Offline:** start FastAPI and check its terminal output; API docs should load at `/docs`.
- **CORS error:** add the exact Vite origin to backend `ALLOWED_ORIGINS`, then restart FastAPI.
- **Wrong class metadata:** verify training label order matches the six required classes; do not reorder the display mapping to conceal mismatched weights.
- **Slow first inference:** CPU inference is supported; speed depends on hardware and model size. CUDA is optional.
- The UI/API currently process one image per request. Detection quality depends on the final trained model and its training data; the repository does not contain that model or a sample dataset.

## Validation status

The backend suite runs without the final model or a GPU; it skips one optional real-model smoke test unless both the weights and a sample image are available. To run that real inference check, set `PCB_SMOKE_IMAGE` to a local sample image before running pytest (`$env:PCB_SMOKE_IMAGE='C:\path\to\board.png'` in PowerShell, or `PCB_SMOKE_IMAGE=/path/to/board.png` on Linux/macOS). The frontend production build and npm dependency audit were run during project setup. Model class compatibility and prediction quality still need validation with the final weights.
