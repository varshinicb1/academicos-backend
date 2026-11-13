# academicos-backend (retired mirror)

This repository is a retired mirror of the AcademicOS backend source, built for the old Render
deployment. Render is no longer used (owner decision, 2026-10-13). Production runs on Google Cloud
Run and Firebase Hosting, built from the main AcademicOS repository.

- Nothing syncs here any more: AcademicOS removed its sync workflow in #145.
- The keep-warm and Supabase keep-alive pings were removed, because they kept a retired service
  awake and the Supabase ping failed every day.
- `build-push.yml` is kept only so the owner can decide whether to archive this repository.
