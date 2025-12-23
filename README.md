# Insight Engine (Intelligent Meeting Insights Platform)

**Insight Engine** is an AI-powered platform designed to perform deep analysis on meeting transcripts, extract action items, summarize discussions, and provide intelligent insights using advanced NLP.

## 🚀 Key Features

### 🧠 Core AI & NLP (`app/nlp_analyzer.py`)
*   **Meeting Summarization**: Automatically generates concise summaries of long discussions.
*   **Action Item Extraction**: Identifies and assigns tasks, owners, and due dates from transcript text.
*   **Key Topic Detection**: Extracts valid topics and decision points.
*   **Speech-to-Text**: Integrated audio processing for meeting recordings.

### 🔒 Security & Auth (`app/auth.py`, `app/password_policy.py`)
*   **Robust Authentication**: JWT-based auth with refresh tokens.
*   **Advanced Password Policy**: Enforces minimum entropy, checks **Have I Been Pwned** (HIBP) for breached passwords, and prevents common weak passwords.
*   **Brute Force Protection**: Rate limiting and lockout mechanisms.
*   **Secrets Management**: Production-ready secrets loading from:
    *   Environment Variables (`.env`)
    *   AWS Secrets Manager
    *   Azure Key Vault
    *   HashiCorp Vault

### 💻 Modern Frontend (`frontend/`)
*   Built with **React 18** and **Vite**.
*   **TailwindCSS** for styling.
*   **Recharts** for data visualization.
*   **Playwright** for End-to-End (E2E) testing.

### ⚙️ Backend Architecture
*   **FastAPI**: High-performance async Python framework.
*   **MongoDB (Motor)**: Async database driver for scalable data storage.
*   **OpenAI Integration**: Leverages LLMs for semantic analysis.

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Backend** | Python 3.x, FastAPI, Uvicorn |
| **Database** | MongoDB, Motor (Async) |
| **AI/ML** | OpenAI API, Faster-Whisper, Vosk |
| **Frontend** | React, Vite, TailwindCSS |
| **Testing** | Pytest, Playwright |
| **Security** | Python-Jose, Passlib, Bcrypt |

## 📦 Installation & Setup

### Prerequisites
*   Python 3.8+
*   Node.js 18+
*   MongoDB instance

### 1. Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.sample .env
# Edit .env with your keys (OPENAI_API_KEY, SECRET_KEY, MONGO_URI)

# Run server
python main.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🧪 Testing
```bash
# Backend Tests
pytest tests/

# Frontend E2E Tests
cd frontend
npx playwright test
```

## 📂 Project Structure
*   `app/`: Core backend logic (API, Auth, NLP).
*   `frontend/`: React application.
*   `tools/`: Admin utilities (Backups, PDF extraction).
*   `tests/`: Integration and unit tests (ignored in git).
