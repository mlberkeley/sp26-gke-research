# Cloud Infrastructure

Terraform for GCP/GKE lives in this directory.

Use Make targets only; GCP auth/setup is automated.

Current project scope in this repo: Track 3+4 research agent runtime (self-healing, persistence, and snapshot/pause-resume exploration).

Secrets workflow uses SOPS:

1. Non-sensitive Terraform config goes in `cloud/config.auto.tfvars` (copy from `cloud/config.auto.tfvars.example`).
2. Optional: copy `secrets/.sops.yaml.example` to `secrets/.sops.yaml` and set a GCP KMS key path.
3. `make tf-secrets-template`
4. Fill `secrets/secrets.sops.yaml` (sensitive values only; keep `dummy_secret` if no real secret exists)
5. Bootstrap KMS key + IAM once: `make gcp-kms-bootstrap`
6. Encrypt: `make tf-secrets-encrypt-kms`
7. Plan/apply: `make tf-plan` then `make tf-apply`

`project_id` is resolved from `PROJECT_ID=...`, then `cloud/config.auto.tfvars`, then active gcloud config.
Default node disk settings are `node_disk_type = "pd-standard"` and `node_disk_size_gb = 30` to reduce SSD quota pressure.
For shared decryption, grant team members `roles/cloudkms.cryptoKeyDecrypter` on the KMS key.
If `secrets/secrets.sops.yaml` is plaintext, Terraform runs with empty secrets (ADC-only).

Main commands:

1. `make gcp-init`
2. `make tf-plan`
3. `make tf-apply`

KMS admin helper:

1. Put CSV emails in `.env` as `KMS_GRANT_MEMBERS_CSV=email1,email2,...`
2. Run `make gcp-admin-kms-setup`

Headless auth:

1. `make GCLOUD_LOGIN_FLAGS=--no-launch-browser gcp-init`

Dummy workload starter on GKE:

1. `make gke-dummy-build`
2. `make gke-dummy-push` (auto-auths gcloud/ADC, configures Docker auth, and creates Artifact Registry repo if needed)
3. `make gke-dummy-run-once` (or `make gke-dummy-schedule` for CronJob)
4. `make gke-dummy-logs`

For this starter, plain manifests are intentional. Move to Helm once you need environment-specific values, chart versioning, or multiple deployable workloads.
If `gke-gcloud-auth-plugin` is missing, Make falls back to short-lived access-token auth for kubectl.

Track 3+4 implementation notes:

1. Keep the dummy workflow as a deployment smoke test while integrating the real research loop.
2. Add persistent checkpoint storage before failure-drill testing.
3. Add explicit restart/retry policy and verify resume behavior by deleting the active pod during a run.
