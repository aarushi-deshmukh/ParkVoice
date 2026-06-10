# Production Deployment

This platform is a screening support system for research use. It is not a diagnostic medical device.

## Frontend - Vercel

1. Set project root to `frontend`.
2. Set `VITE_API_BASE_URL` to the Railway backend URL.
3. Build with `npm run build`.
4. Serve `dist`.

## Backend - Railway

1. Set project root to `backend`.
2. Use the included `Dockerfile` and `railway.json`.
3. Set environment variables from `backend/.env.example`.
4. Set `DATABASE_URL` to Neon PostgreSQL.
5. Set Supabase storage credentials if uploaded audio should be stored outside the Railway filesystem.

## Database - Neon PostgreSQL

Use a pooled or direct Neon connection string. `postgres://` and `postgresql://` URLs are normalized to `postgresql+asyncpg://` at runtime.

## Storage - Supabase Storage

Create a private bucket such as `voice-recordings`. Keep audio access private and return signed URLs only in authenticated deployments.

## Benchmarks

Run `python backend/evaluation/edge_benchmark.py` on each real target machine you intend to report. Do not publish Raspberry Pi or Jetson results unless the script was executed on that hardware.
