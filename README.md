# CFAR Demonstrator

A Streamlit teaching app for comparing CA-CFAR, GOCA-CFAR, and SOCA-CFAR under multi-target and clutter-edge scenarios. It includes an automatic Monte Carlo experiment module with dynamic visualization and cumulative performance statistics.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py`: main Streamlit application
- `requirements.txt`: Python dependencies for cloud deployment
- `.streamlit/config.toml`: basic Streamlit theme and server configuration

## Deployment

Recommended platform: Streamlit Community Cloud.

1. Create a GitHub repository.
2. Upload all files in this folder to the repository root.
3. Open Streamlit Community Cloud and create a new app from the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.
