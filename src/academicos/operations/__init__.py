"""School operations that change every day: notifications (M3) and homework
(M5). One SQLite file, snapshotted to GCS like the curriculum
(storage/snapshot_sync.py), under its own key so its churn never re-uploads
the curriculum."""
