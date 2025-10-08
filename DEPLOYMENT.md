# Deployment (Railway)

This app is ready for Railway using `uvicorn` and an ASGI wrapper.

## Prerequisites
- Railway account and CLI (optional for local)
- Landing AI API key

## Environment Variables
- `LANDING_AI_API_KEY`: Landing AI key
- `FLASK_SECRET_KEY`: session secret
- `PORT`: provided by Railway automatically

## Files
- `Procfile`:

```
web: uvicorn app:asgi_app --host 0.0.0.0 --port ${PORT:-8000}
```

- `Dockerfile`: includes WeasyPrint system dependencies

## Steps
1. Push repository to GitHub.
2. Create a new Railway project, deploy from repo.
3. Add variables in Settings → Variables.
4. Deploy. Railway will build the Docker image and run `uvicorn`.

## Notes
- If WeasyPrint is unavailable in your platform, the export endpoint will return HTML for manual printing as PDF.
- Scale settings and logs are available in Railway dashboard.
