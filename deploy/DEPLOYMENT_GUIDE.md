# Quick Deployment Guide for RealEstate AI Assistant
Get a working public HTTPS URL to submit to DarGlobal in under 5 minutes.

---

## Option 1: Render.com (Recommended Free Cloud URL)
Render provides free Docker web services directly connected to your GitHub repository.

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "add realestate ai assistant"
   git remote add origin https://github.com/<your-username>/realestate-ai-assistant.git
   git branch -M main
   git push -u origin main
   ```
2. **Deploy on Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com/) (Sign in with GitHub).
   - Click **New +** -> **Web Service**.
   - Select your repository `realestate-ai-assistant`.
   - Set **Runtime**: `Docker`.
   - Under **Environment Variables**, add:
     - `OPENROUTER_API_KEY`: `<your_openrouter_api_key>`
     - `PORT`: `8000`
   - Click **Create Web Service**.
3. **Result**:
   Within 3 minutes, Render will build your Docker image and provide a live URL:
   `https://realestate-ai-assistant.onrender.com`

---

## Option 2: Hugging Face Spaces (Instant Free Docker Space)
Hugging Face Spaces gives free CPU Docker containers with zero cold-start limits.

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Space name: `realestate-ai-assistant`
3. Space SDK: Select **Docker** -> **Blank**.
4. In your Space's Settings -> **Variables and secrets**, add secret:
   - Name: `OPENROUTER_API_KEY`, Value: `<your_openrouter_api_key>`
5. Clone the space repo and push your code:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-hf-username>/realestate-ai-assistant
   git push hf main
   ```
6. **Result**:
   Your live working URL will be:
   `https://<your-hf-username>-realestate-ai-assistant.hf.space`

---

## Option 3: Instant Local Tunnel (Cloudflare or ngrok - 30 Seconds)
If you want an immediate live HTTPS URL right from your laptop without uploading to cloud hosting:

### Using Cloudflare Tunnel (No account needed):
1. Download `cloudflared.exe` or run:
   ```powershell
   winget install --id Cloudflare.cloudflared
   ```
2. Start the local server:
   ```powershell
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
3. In another terminal, run:
   ```powershell
   cloudflared tunnel --url http://localhost:8000
   ```
4. Cloudflare will print a live public HTTPS URL (e.g. `https://random-words.trycloudflare.com`). Anyone worldwide can test your chatbot directly!

---

## Option 4: Local Docker Run
To verify the Docker container locally:
```bash
# Build the Docker image
docker build -t realestate-ai-assistant .

# Run the container
docker run -p 8000:8000 --env-file .env realestate-ai-assistant

# Access at http://localhost:8000
```
