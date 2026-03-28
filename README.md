# remop

Monorepo for the remop project: a Next.js frontend and a Python backend for perception experiments (Ultralytics YOLO26 plus MiDaS monocular depth).

## Repository layout

| Path | Description |
|------|-------------|
| [`frontend-remop/`](frontend-remop/) | Next.js 16 app (React 19, Tailwind CSS 4) |
| [`backend/`](backend/) | Python tools: real-time webcam detection + relative depth (`yolo26_depth_webcam.py`) |

## Frontend

```bash
cd frontend-remop
npm install
npm run dev
```

Open the URL shown in the terminal (typically [http://localhost:3000](http://localhost:3000)).

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python yolo26_depth_webcam.py
```

Press `q` to quit the OpenCV window. See the script docstring for flags (`--model`, `--source`, `--no-depth-panel`).

**Note:** First run downloads YOLO26 and MiDaS weights; `timm` is pinned in `requirements.txt` for MiDaS v3.1 compatibility.

## License

This project is licensed under the MIT License—see [LICENSE](LICENSE).
