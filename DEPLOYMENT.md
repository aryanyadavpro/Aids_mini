# 🚀 ResoGenre — Complete Cloud Deployment Guide

> **Deploy your Deep-Learning Audio Genre Classification System to production for FREE.**  
> Architecture: **FastAPI + PyTorch Backend** (Docker on Render/Railway/HuggingFace) + **Next.js 16 Frontend** (Vercel).

---

## 📌 Architecture Overview for Deployment

```
[ User Browser ]
       │
       ▼
[ Frontend: Vercel (100% Free) ]
  • URL: https://resogenre.vercel.app
  • Framework: Next.js 16 + React 19
  • Env Variable: NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
       │
       │ (REST API Requests: /predict, /waveform, /spectrogram)
       ▼
[ Backend: Render.com / Railway / Hugging Face Spaces (Free) ]
  • URL: https://your-backend.onrender.com
  • Container: Docker (Python 3.11-slim + CPU PyTorch + Librosa + FastAPI)
  • Model: Pre-packaged models/resogenre_cnn_best.pth (5.1 MB)
```

---

## 🌟 Method 1: The Recommended Free Cloud Setup (Vercel + Render)

### Step 1: Push Code to GitHub

Open terminal in `c:\AIDS_mini` and push your code:
```bash
git add .
git commit -m "feat: complete resogenre full-stack audio classification system"
git push origin main
```

*(Note: `.gitignore` has already been configured to include your best model `models/resogenre_cnn_best.pth` while excluding huge raw audio files and node_modules).*

---

### Step 2: Deploy Backend on Render.com (Free)

1. Sign up / Log in to [Render.com](https://render.com).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository (`AIDS_mini`).
4. Configure the service settings:
   - **Name:** `resogenre-backend`
   - **Region:** Any (e.g., Oregon or Frankfurt)
   - **Branch:** `main`
   - **Runtime:** **Docker** *(Render will automatically detect `Dockerfile` in the root!)*
   - **Instance Type:** **Free** (512 MB RAM, CPU)
5. Click **Create Web Service**.
6. Render will build the Docker container (takes ~3-4 minutes) and provide a live URL, for example:
   ```
   https://resogenre-backend.onrender.com
   ```
7. Test it by opening in your browser:
   `https://resogenre-backend.onrender.com/health`  
   *(Should return: `{"status": "healthy", "model_loaded": true, "device": "cpu"}`)*

---

### Step 3: Deploy Frontend on Vercel (Free)

1. Sign up / Log in to [Vercel.com](https://vercel.com).
2. Click **Add New...** -> **Project**.
3. Import your GitHub repository (`AIDS_mini`).
4. In the **Configure Project** screen:
   - **Framework Preset:** Next.js *(Auto-detected)*
   - **Root Directory:** Click **Edit** and select **`frontend`** folder.
5. Under **Environment Variables**, add:
   - **Key:** `NEXT_PUBLIC_API_URL`
   - **Value:** Your Render backend URL from Step 2 (e.g. `https://resogenre-backend.onrender.com`)
   *(Make sure there is no trailing slash `/` at the end)*
6. Click **Deploy**.
7. In ~60 seconds, your frontend will be live at:
   ```
   https://resogenre.vercel.app
   ```

🎉 **Your full-stack application is now live on the internet!**

---

## 🚂 Method 2: Deploy Backend on Railway.app (Alternative to Render)

Railway provides $5 monthly free credits and starts up instantly without sleeping:
1. Go to [Railway.app](https://railway.app) and login with GitHub.
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Select your `AIDS_mini` repository.
4. Railway will automatically detect the root `Dockerfile`.
5. Under **Settings** -> **Networking**, click **Generate Domain**.
6. Copy the generated domain (e.g. `https://resogenre-production.up.railway.app`) and use it as `NEXT_PUBLIC_API_URL` in Vercel.

---

## 🤗 Method 3: Deploy on Hugging Face Spaces (16GB RAM Free for ML!)

Hugging Face Spaces is designed specifically for machine learning models and gives **16 GB RAM + 2 vCPU for free**:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Space Name: `resogenre-api`
3. Space SDK: Select **Docker** -> **Blank**.
4. Clone the space locally or push the repository files:
   - `Dockerfile`
   - `requirements.txt`
   - `config.py`
   - `src/`
   - `api/`
   - `models/resogenre_cnn_best.pth`
5. Hugging Face builds your Docker container. Your API endpoint will be:
   `https://<username>-resogenre-api.hf.space`

---

## 🐳 Method 4: Full-Stack Docker Deployment (Local or VPS)

If you have a Linux VPS (DigitalOcean, AWS EC2, Linode) or want to run everything in containers locally:

```bash
# Clone the repository
git clone https://github.com/<your-username>/AIDS_mini.git
cd AIDS_mini

# Build and start both containers in background
docker compose up -d --build
```

- **Frontend:** `http://<your-server-ip>:3000`
- **Backend:** `http://<your-server-ip>:8000`
- **Swagger Docs:** `http://<your-server-ip>:8000/docs`

---

## 🛠️ Verification Checklist After Deployment

- [ ] Open `<backend-url>/health` in browser: returns `{"status":"healthy","model_loaded":true}`.
- [ ] Open `<frontend-url>` in browser: Neo-brutalist UI loads with Space Grotesk fonts.
- [ ] Click the **Jazz** or **Rock** 1-click test button: audio plays and predicted genre appears within seconds.
- [ ] Upload an MP3/WAV file: waveform canvas animates and Mel-Spectrogram appears.
