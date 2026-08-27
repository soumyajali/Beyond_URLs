# Beyond URLs - Deployment Guide

This project is a Flask-based web application that utilizes Machine Learning libraries (PyTorch and Transformers) to analyze text for financial phishing.

Because this application relies on heavy Machine Learning models, **it requires special consideration for deployment.**

## ⚠️ Important Note About Serverless Deployment (Vercel/Netlify)

Platforms like **Vercel** and **Netlify** use AWS Lambda Serverless Functions under the hood. These functions have a strict **250 MB size limit** for all code and dependencies combined. 

Since `torch` (PyTorch) is often over 700 MB, **deploying this app to Vercel will likely fail during the build process** due to exceeding size limits.

If you still wish to attempt Vercel, the required `pyproject.toml` entrypoint configuration has already been added to this repository. However, it is highly recommended to use a container-based or VM-based hosting provider as described below.

---

## 🚀 Recommended Deployment: Render.com (or Railway.app)

Platforms like [Render](https://render.com) and [Railway](https://railway.app) do not have the strict 250MB size limitations and behave like traditional servers, making them perfect for PyTorch/Flask applications.

### Deploying to Render.com

1. **Create an Account:** Go to [Render.com](https://render.com) and sign up using your GitHub account.
2. **New Web Service:** Click on "New +" at the top right and select **Web Service**.
3. **Connect Repository:** Select this repository (`Beyond_URLs`) from your linked GitHub account.
4. **Configure the Service:**
   - **Name:** Choose a name (e.g., `beyond-urls-app`).
   - **Environment:** `Python`
   - **Region:** Choose whatever is closest to you.
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app.api:app --bind 0.0.0.0:$PORT` (This tells Render to start the Flask app using Gunicorn and bind to the correct port).
5. **Select Tier:** Select the Free tier (or a paid tier if you need more RAM for the PyTorch model to run quickly).
6. **Deploy:** Click **Create Web Service**. Render will automatically build the environment, install the massive PyTorch libraries, and start your app.

---

## 💻 Local Development

If you just want to run the app on your own computer:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/soumyajali/Beyond_URLs.git
   cd Beyond_URLs
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   export FLASK_APP=app/api.py  # On Windows CMD use: set FLASK_APP=app/api.py
   flask run
   ```
   *Alternatively, you can just run `python app/api.py`.*

5. **Open your browser:** Navigate to `http://localhost:8431` (or whatever port Flask indicates in the terminal).
