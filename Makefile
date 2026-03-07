TF_DIR := cloud
SECRETS_DIR := secrets
TF_PLAN_FILE := tfplan
TF_CONFIG_FILE := $(TF_DIR)/config.auto.tfvars
TF_CONFIG_TEMPLATE := $(TF_DIR)/config.auto.tfvars.example
TF_SECRETS_FILE := $(SECRETS_DIR)/secrets.sops.yaml
TF_SECRETS_TEMPLATE := $(SECRETS_DIR)/secrets.sops.yaml.example
TF_SECRETS_JSON := $(SECRETS_DIR)/.secrets.auto.tfvars.json

# Load local environment overrides (ignored by git).
-include .env
export

GCLOUD_LOGIN_FLAGS ?=
PROJECT_ID ?= cogent-nimbus-489503-f6
SOPS_GCP_KMS ?=
SOPS_AGE_RECIPIENT ?=
SOPS_KMS_LOCATION ?= global
SOPS_KMS_KEYRING ?= sops
SOPS_KMS_KEY ?= infra-secrets
GCLOUD_CONFIG_DIR ?= .gcloud
ADMIN_GCLOUD_CONFIG_DIR ?= .gcloud-admin
GCP_REQUIRED_SERVICES := compute.googleapis.com container.googleapis.com cloudkms.googleapis.com artifactregistry.googleapis.com
KMS_GRANT_MEMBER ?=
KMS_GRANT_MEMBERS_CSV ?= robin.holzinger@berkeley.edu
GKE_CLUSTER_NAME ?= research-agent
GKE_CLUSTER_REGION ?= us-central1
GKE_NAMESPACE ?= default
GKE_DUMMY_AR_REGION ?= us-central1
GKE_DUMMY_AR_REPOSITORY ?= gke-workflows
GKE_DUMMY_IMAGE ?= $(GKE_DUMMY_AR_REGION)-docker.pkg.dev/$(PROJECT_ID)/$(GKE_DUMMY_AR_REPOSITORY)/gke-dummy-workflow:latest
GKE_DUMMY_DOCKERFILE ?= cloud/docker/gke-dummy.Dockerfile
GKE_DUMMY_DOCKER_PLATFORM ?= linux/amd64
GKE_DUMMY_MANIFEST_DIR ?= cloud/k8s/dummy-workflow

GCLOUD_CONFIG_ABS := $(abspath $(GCLOUD_CONFIG_DIR))
ADMIN_GCLOUD_CONFIG_ABS := $(abspath $(ADMIN_GCLOUD_CONFIG_DIR))
ADC_FILE := $(GCLOUD_CONFIG_ABS)/application_default_credentials.json

TF := CLOUDSDK_CONFIG=$(GCLOUD_CONFIG_ABS) GOOGLE_APPLICATION_CREDENTIALS=$(ADC_FILE) pixi run terraform -chdir=$(TF_DIR)
SOPS := CLOUDSDK_CONFIG=$(GCLOUD_CONFIG_ABS) GOOGLE_APPLICATION_CREDENTIALS=$(ADC_FILE) pixi run sops
GCLOUD := CLOUDSDK_CONFIG=$(GCLOUD_CONFIG_ABS) pixi run gcloud
ADMIN_GCLOUD := CLOUDSDK_CONFIG=$(ADMIN_GCLOUD_CONFIG_ABS) pixi run gcloud
KUBECTL := CLOUDSDK_CONFIG=$(GCLOUD_CONFIG_ABS) pixi run kubectl

.PHONY: clean \
	tf-init tf-fmt tf-validate tf-plan tf-apply tf-destroy tf-output \
	tf-config-template tf-config-check \
	tf-secrets-template tf-secrets-encrypt tf-secrets-encrypt-kms tf-secrets-edit tf-secrets-decrypt tf-secrets-materialize tf-secrets-clean \
	gcp-auth gcp-project gcp-adc-quota gcp-enable-services gcp-kms-bootstrap gcp-init gcp-docker-auth gcp-artifact-registry-repo \
	gcp-admin-auth gcp-admin-project gcp-admin-kms-create-keyring gcp-admin-kms-create-key gcp-admin-kms-grant-user gcp-admin-kms-setup \
	gke-auth gke-namespace gke-dummy-build gke-dummy-push gke-dummy-run-once gke-dummy-schedule gke-dummy-delete gke-dummy-logs \
	logout

# ------------------------------------------------------------------------------------ #
#                                       Terraform                                      #
# ------------------------------------------------------------------------------------ #

tf-init:
	$(TF) init

tf-fmt:
	$(TF) fmt -recursive

tf-validate: tf-init
	$(TF) validate

tf-plan: tf-validate tf-config-check gcp-init tf-secrets-materialize
	@PROJECT_ID_RESOLVED="$$( $(GCLOUD) config get-value project 2>/dev/null | tr -d '\n' )"; \
	test -n "$$PROJECT_ID_RESOLVED" && [ "$$PROJECT_ID_RESOLVED" != "(unset)" ] || (echo "No gcloud project is configured."; exit 1); \
	$(TF) plan -var "project_id=$$PROJECT_ID_RESOLVED" -var-file=$(abspath $(TF_CONFIG_FILE)) -var-file=$(abspath $(TF_SECRETS_JSON)) -out=$(TF_PLAN_FILE)

tf-apply: tf-plan
	echo "Applying Terraform plan from $(TF_PLAN_FILE)..."
	echo "Starting a new GKE cluster can take several minutes. Please be patient and do not interrupt the process."
	$(TF) apply $(TF_PLAN_FILE)

tf-destroy: tf-validate tf-config-check gcp-init tf-secrets-materialize
	@PROJECT_ID_RESOLVED="$$( $(GCLOUD) config get-value project 2>/dev/null | tr -d '\n' )"; \
	test -n "$$PROJECT_ID_RESOLVED" && [ "$$PROJECT_ID_RESOLVED" != "(unset)" ] || (echo "No gcloud project is configured."; exit 1); \
	$(TF) destroy -var "project_id=$$PROJECT_ID_RESOLVED" -var-file=$(abspath $(TF_CONFIG_FILE)) -var-file=$(abspath $(TF_SECRETS_JSON))

tf-output:
	$(TF) output

tf-config-check:
	@test -f "$(TF_CONFIG_FILE)" || (echo "Missing $(TF_CONFIG_FILE). Run make tf-config-template first."; exit 1)

tf-secrets-encrypt:
	@test -f "$(TF_SECRETS_FILE)" || (echo "Missing $(TF_SECRETS_FILE). Run make tf-secrets-template first."; exit 1)
	@if [ -n "$(SOPS_GCP_KMS)" ]; then \
		$(SOPS) --encrypt --gcp-kms "$(SOPS_GCP_KMS)" --in-place "$(TF_SECRETS_FILE)"; \
	elif [ -n "$(SOPS_AGE_RECIPIENT)" ]; then \
		$(SOPS) --encrypt --age "$(SOPS_AGE_RECIPIENT)" --in-place "$(TF_SECRETS_FILE)"; \
	elif [ -f "$(SECRETS_DIR)/.sops.yaml" ]; then \
		$(SOPS) --encrypt --in-place "$(TF_SECRETS_FILE)"; \
	else \
		echo "Set SOPS_GCP_KMS=projects/.../cryptoKeys/... or SOPS_AGE_RECIPIENT=age1..., or create $(SECRETS_DIR)/.sops.yaml"; \
		exit 1; \
	fi

tf-secrets-encrypt-kms: gcp-auth gcp-project
	@test -f "$(TF_SECRETS_FILE)" || (echo "Missing $(TF_SECRETS_FILE). Run make tf-secrets-template first."; exit 1)
	@PROJECT_ID_RESOLVED="$$( $(GCLOUD) config get-value project 2>/dev/null | tr -d '\n' )"; \
	test -n "$$PROJECT_ID_RESOLVED" && [ "$$PROJECT_ID_RESOLVED" != "(unset)" ] || (echo "No gcloud project is configured."; exit 1); \
	KMS_PATH="projects/$$PROJECT_ID_RESOLVED/locations/$(SOPS_KMS_LOCATION)/keyRings/$(SOPS_KMS_KEYRING)/cryptoKeys/$(SOPS_KMS_KEY)"; \
	echo "Encrypting $(TF_SECRETS_FILE) with $$KMS_PATH"; \
	$(SOPS) --encrypt --gcp-kms "$$KMS_PATH" --in-place "$(TF_SECRETS_FILE)" || ( \
		echo ""; \
		echo "KMS encryption failed. Run: make gcp-kms-bootstrap"; \
		echo "Then retry: make tf-secrets-encrypt-kms"; \
		exit 1; \
	)

tf-secrets-edit:
	@test -f "$(TF_SECRETS_FILE)" || (echo "Missing $(TF_SECRETS_FILE)."; exit 1)
	$(SOPS) "$(TF_SECRETS_FILE)"

tf-secrets-decrypt:
	@test -f "$(TF_SECRETS_FILE)" || (echo "Missing $(TF_SECRETS_FILE)."; exit 1)
	@if grep -q '^[[:space:]]*sops:[[:space:]]*$$' "$(TF_SECRETS_FILE)"; then \
		$(SOPS) --decrypt --output-type json "$(TF_SECRETS_FILE)"; \
	else \
		echo "{}"; \
	fi

tf-secrets-materialize:
	@test -f "$(TF_SECRETS_FILE)" || (echo "Missing $(TF_SECRETS_FILE)."; exit 1)
	@if grep -q '^[[:space:]]*sops:[[:space:]]*$$' "$(TF_SECRETS_FILE)"; then \
		$(SOPS) --decrypt --output-type json "$(TF_SECRETS_FILE)" > "$(TF_SECRETS_JSON)"; \
	else \
		echo "{}" > "$(TF_SECRETS_JSON)"; \
	fi

tf-secrets-clean:
	rm -f "$(TF_SECRETS_JSON)"

# ------------------------------------------------------------------------------------ #
#                              GCP (user context: .gcloud)                             #
# ------------------------------------------------------------------------------------ #

gcp-auth:
	@$(GCLOUD) auth print-access-token >/dev/null 2>&1 || (echo "Running gcloud auth login..."; $(GCLOUD) auth login $(GCLOUD_LOGIN_FLAGS))
	@$(GCLOUD) auth application-default print-access-token >/dev/null 2>&1 || (echo "Running gcloud auth application-default login..."; $(GCLOUD) auth application-default login $(GCLOUD_LOGIN_FLAGS))

gcp-project:
	@CONFIG_PROJECT_ID=""; \
	if [ -f "$(TF_CONFIG_FILE)" ]; then \
		CONFIG_PROJECT_ID="$$(sed -n 's/^[[:space:]]*project_id[[:space:]]*=[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' "$(TF_CONFIG_FILE)" | head -n 1)"; \
	fi; \
	DEFAULT_PROJECT_ID="$(PROJECT_ID)"; \
	CURRENT_PROJECT_ID="$$( $(GCLOUD) config get-value project 2>/dev/null | tr -d '\n' )"; \
	if [ "$$CURRENT_PROJECT_ID" = "(unset)" ]; then CURRENT_PROJECT_ID=""; fi; \
	PROJECT_ID_RESOLVED="$$DEFAULT_PROJECT_ID"; \
	if [ -z "$$PROJECT_ID_RESOLVED" ]; then PROJECT_ID_RESOLVED="$$CONFIG_PROJECT_ID"; fi; \
	if [ -z "$$PROJECT_ID_RESOLVED" ]; then PROJECT_ID_RESOLVED="$$CURRENT_PROJECT_ID"; fi; \
	if [ -z "$$PROJECT_ID_RESOLVED" ]; then \
		echo "Set PROJECT_ID=... (for example: make PROJECT_ID=my-project tf-plan)."; \
		exit 1; \
	fi; \
	$(GCLOUD) config set project "$$PROJECT_ID_RESOLVED"

gcp-adc-quota: gcp-project
	@PROJECT_ID_RESOLVED="$$( $(GCLOUD) config get-value project 2>/dev/null | tr -d '\n' )"; \
	test -n "$$PROJECT_ID_RESOLVED" && [ "$$PROJECT_ID_RESOLVED" != "(unset)" ] || (echo "No gcloud project is configured."; exit 1); \
	$(GCLOUD) auth application-default set-quota-project "$$PROJECT_ID_RESOLVED"

gcp-enable-services: gcp-adc-quota
	@PROJECT_ID_RESOLVED="$$( $(GCLOUD) config get-value project 2>/dev/null | tr -d '\n' )"; \
	test -n "$$PROJECT_ID_RESOLVED" && [ "$$PROJECT_ID_RESOLVED" != "(unset)" ] || (echo "No gcloud project is configured."; exit 1); \
	$(GCLOUD) services enable $(GCP_REQUIRED_SERVICES) --project "$$PROJECT_ID_RESOLVED"

gcp-kms-bootstrap: gcp-enable-services
	@PROJECT_ID_RESOLVED="$$( $(GCLOUD) config get-value project 2>/dev/null | tr -d '\n' )"; \
	test -n "$$PROJECT_ID_RESOLVED" && [ "$$PROJECT_ID_RESOLVED" != "(unset)" ] || (echo "No gcloud project is configured."; exit 1); \
	ACCOUNT="$$( $(GCLOUD) config get-value account 2>/dev/null | tr -d '\n' )"; \
	test -n "$$ACCOUNT" && [ "$$ACCOUNT" != "(unset)" ] || (echo "No active gcloud account configured."; exit 1); \
	if echo "$$ACCOUNT" | grep -q 'gserviceaccount.com$$'; then MEMBER="serviceAccount:$$ACCOUNT"; else MEMBER="user:$$ACCOUNT"; fi; \
	($(GCLOUD) kms keyrings describe "$(SOPS_KMS_KEYRING)" --location "$(SOPS_KMS_LOCATION)" >/dev/null 2>&1 || $(GCLOUD) kms keyrings create "$(SOPS_KMS_KEYRING)" --location "$(SOPS_KMS_LOCATION)") && \
	($(GCLOUD) kms keys describe "$(SOPS_KMS_KEY)" --keyring "$(SOPS_KMS_KEYRING)" --location "$(SOPS_KMS_LOCATION)" >/dev/null 2>&1 || $(GCLOUD) kms keys create "$(SOPS_KMS_KEY)" --keyring "$(SOPS_KMS_KEYRING)" --location "$(SOPS_KMS_LOCATION)" --purpose "encryption") && \
	$(GCLOUD) kms keys add-iam-policy-binding "$(SOPS_KMS_KEY)" --keyring "$(SOPS_KMS_KEYRING)" --location "$(SOPS_KMS_LOCATION)" --member "$$MEMBER" --role "roles/cloudkms.cryptoKeyEncrypterDecrypter" && \
	echo "KMS key ready and IAM binding applied for $$MEMBER."

gcp-init: gcp-auth gcp-enable-services
	@echo "GCP auth and project setup ready."

gcp-docker-auth: gcp-init
	@REGISTRY_HOST="$$(echo "$(GKE_DUMMY_IMAGE)" | cut -d/ -f1)"; \
	echo "Configuring Docker auth for $$REGISTRY_HOST"; \
	$(GCLOUD) auth configure-docker "$$REGISTRY_HOST" --quiet

gcp-artifact-registry-repo: gcp-init
	@PROJECT_ID_RESOLVED="$$( $(GCLOUD) config get-value project 2>/dev/null | tr -d '\n' )"; \
	test -n "$$PROJECT_ID_RESOLVED" && [ "$$PROJECT_ID_RESOLVED" != "(unset)" ] || (echo "No gcloud project is configured."; exit 1); \
	($(GCLOUD) artifacts repositories describe "$(GKE_DUMMY_AR_REPOSITORY)" --location "$(GKE_DUMMY_AR_REGION)" --project "$$PROJECT_ID_RESOLVED" >/dev/null 2>&1 || \
	$(GCLOUD) artifacts repositories create "$(GKE_DUMMY_AR_REPOSITORY)" --repository-format=docker --location "$(GKE_DUMMY_AR_REGION)" --description "GKE workflow images" --project "$$PROJECT_ID_RESOLVED")

# ------------------------------------------------------------------------------------ #
#                                  GKE Dummy Workflow                                  #
# ------------------------------------------------------------------------------------ #

gke-auth: gcp-init
	@PLUGIN_PATH="$$(command -v gke-gcloud-auth-plugin || true)"; \
	if [ -n "$$PLUGIN_PATH" ]; then \
		$(GCLOUD) container clusters get-credentials "$(GKE_CLUSTER_NAME)" --region "$(GKE_CLUSTER_REGION)" --project "$(PROJECT_ID)"; \
	else \
		echo "gke-gcloud-auth-plugin not found; configuring kubectl with short-lived access token."; \
		ENDPOINT="$$( $(GCLOUD) container clusters describe "$(GKE_CLUSTER_NAME)" --region "$(GKE_CLUSTER_REGION)" --project "$(PROJECT_ID)" --format='value(endpoint)' )"; \
		CA_CERT="$$( $(GCLOUD) container clusters describe "$(GKE_CLUSTER_NAME)" --region "$(GKE_CLUSTER_REGION)" --project "$(PROJECT_ID)" --format='value(masterAuth.clusterCaCertificate)' )"; \
		test -n "$$ENDPOINT" || (echo "Failed to resolve cluster endpoint."; exit 1); \
		test -n "$$CA_CERT" || (echo "Failed to resolve cluster CA certificate."; exit 1); \
		CTX="gke_$(PROJECT_ID)_$(GKE_CLUSTER_REGION)_$(GKE_CLUSTER_NAME)"; \
		USER_NAME="token-user-$(GKE_CLUSTER_NAME)"; \
		TOKEN="$$( $(GCLOUD) auth print-access-token )"; \
		$(KUBECTL) config set-cluster "$$CTX" --server="https://$$ENDPOINT" --certificate-authority-data="$$CA_CERT" >/dev/null; \
		$(KUBECTL) config set-credentials "$$USER_NAME" --token="$$TOKEN" >/dev/null; \
		$(KUBECTL) config set-context "$$CTX" --cluster="$$CTX" --user="$$USER_NAME" >/dev/null; \
		$(KUBECTL) config use-context "$$CTX" >/dev/null; \
	fi

gke-namespace: gke-auth
	@$(KUBECTL) get namespace "$(GKE_NAMESPACE)" >/dev/null 2>&1 || $(KUBECTL) create namespace "$(GKE_NAMESPACE)"

gke-dummy-build:
	docker build --platform "$(GKE_DUMMY_DOCKER_PLATFORM)" -f "$(GKE_DUMMY_DOCKERFILE)" -t "$(GKE_DUMMY_IMAGE)" .

gke-dummy-push: gcp-docker-auth gcp-artifact-registry-repo
	@REGISTRY_HOST="$$(echo "$(GKE_DUMMY_IMAGE)" | cut -d/ -f1)"; \
	$(GCLOUD) auth print-access-token | docker login -u oauth2accesstoken --password-stdin "https://$$REGISTRY_HOST"
	@CLOUDSDK_CONFIG=$(GCLOUD_CONFIG_ABS) docker push "$(GKE_DUMMY_IMAGE)"

gke-dummy-run-once: gke-namespace
	@sed -e 's|__IMAGE__|$(GKE_DUMMY_IMAGE)|g' -e 's|__NAMESPACE__|$(GKE_NAMESPACE)|g' "$(GKE_DUMMY_MANIFEST_DIR)/job.yaml" | $(KUBECTL) apply -f -

gke-dummy-schedule: gke-namespace
	@sed -e 's|__IMAGE__|$(GKE_DUMMY_IMAGE)|g' -e 's|__NAMESPACE__|$(GKE_NAMESPACE)|g' "$(GKE_DUMMY_MANIFEST_DIR)/cronjob.yaml" | $(KUBECTL) apply -f -

gke-dummy-delete: gke-auth
	@$(KUBECTL) -n "$(GKE_NAMESPACE)" delete cronjob gke-dummy-workflow --ignore-not-found
	@$(KUBECTL) -n "$(GKE_NAMESPACE)" delete job gke-dummy-workflow-once --ignore-not-found

gke-dummy-logs: gke-auth
	@$(KUBECTL) -n "$(GKE_NAMESPACE)" logs -l app=gke-dummy-workflow --all-containers=true --tail=200 --prefix=true || ( \
		echo "Logs are not available yet (pod likely still creating). Showing pod status/events:"; \
		$(KUBECTL) -n "$(GKE_NAMESPACE)" get pods -l app=gke-dummy-workflow -o wide; \
		POD="$$( $(KUBECTL) -n "$(GKE_NAMESPACE)" get pods -l app=gke-dummy-workflow -o jsonpath='{.items[0].metadata.name}' 2>/dev/null )"; \
		if [ -n "$$POD" ]; then \
			echo ""; \
			$(KUBECTL) -n "$(GKE_NAMESPACE)" describe pod "$$POD" | sed -n '/Events:/,$$p'; \
		fi; \
		true \
	)

# ------------------------------------------------------------------------------------ #
#                                        Session                                       #
# ------------------------------------------------------------------------------------ #

logout:
	@$(GCLOUD) auth revoke --all --quiet || true
	@$(GCLOUD) auth application-default revoke --quiet || true
	@echo "Logged out from gcloud user and ADC credentials in $(GCLOUD_CONFIG_DIR)."

# ------------------------------------------------------------------------------------ #
#                 GCP KMS Admin Helpers (admin context: .gcloud-admin)                 #
# ------------------------------------------------------------------------------------ #

gcp-admin-auth:
	@$(ADMIN_GCLOUD) auth print-access-token >/dev/null 2>&1 || (echo "Running admin gcloud auth login..."; $(ADMIN_GCLOUD) auth login $(GCLOUD_LOGIN_FLAGS))

gcp-admin-project: gcp-admin-auth
	@$(ADMIN_GCLOUD) config set project "$(PROJECT_ID)"

gcp-admin-kms-create-keyring: gcp-admin-project
	@($(ADMIN_GCLOUD) kms keyrings describe "$(SOPS_KMS_KEYRING)" --location "$(SOPS_KMS_LOCATION)" >/dev/null 2>&1 || $(ADMIN_GCLOUD) kms keyrings create "$(SOPS_KMS_KEYRING)" --location "$(SOPS_KMS_LOCATION)")

gcp-admin-kms-create-key: gcp-admin-kms-create-keyring
	@($(ADMIN_GCLOUD) kms keys describe "$(SOPS_KMS_KEY)" --keyring "$(SOPS_KMS_KEYRING)" --location "$(SOPS_KMS_LOCATION)" >/dev/null 2>&1 || $(ADMIN_GCLOUD) kms keys create "$(SOPS_KMS_KEY)" --keyring "$(SOPS_KMS_KEYRING)" --location "$(SOPS_KMS_LOCATION)" --purpose "encryption")

gcp-admin-kms-grant-user: gcp-admin-kms-create-key
	@MEMBER_LIST="$(KMS_GRANT_MEMBER)"; \
	if [ -z "$$MEMBER_LIST" ]; then MEMBER_LIST="$(KMS_GRANT_MEMBERS_CSV)"; fi; \
	test -n "$$MEMBER_LIST" || (echo "Set KMS_GRANT_MEMBER or KMS_GRANT_MEMBERS_CSV."; exit 1); \
	for RAW_MEMBER in $$(printf '%s' "$$MEMBER_LIST" | tr ',' ' '); do \
		MEMBER="$$(echo "$$RAW_MEMBER" | tr -d '[:space:]')"; \
		test -n "$$MEMBER" || continue; \
		if echo "$$MEMBER" | grep -qE '^(user:|serviceAccount:|group:|domain:)'; then \
			MEMBER_ARG="$$MEMBER"; \
		elif echo "$$MEMBER" | grep -q 'gserviceaccount.com$$'; then \
			MEMBER_ARG="serviceAccount:$$MEMBER"; \
		else \
			MEMBER_ARG="user:$$MEMBER"; \
		fi; \
		echo "Granting roles/cloudkms.cryptoKeyEncrypterDecrypter to $$MEMBER_ARG"; \
		$(ADMIN_GCLOUD) kms keys add-iam-policy-binding "$(SOPS_KMS_KEY)" --keyring "$(SOPS_KMS_KEYRING)" --location "$(SOPS_KMS_LOCATION)" --member "$$MEMBER_ARG" --role "roles/cloudkms.cryptoKeyEncrypterDecrypter"; \
	done

gcp-admin-kms-setup: gcp-admin-kms-grant-user
	@MEMBER_LIST="$(KMS_GRANT_MEMBER)"; \
	if [ -z "$$MEMBER_LIST" ]; then MEMBER_LIST="$(KMS_GRANT_MEMBERS_CSV)"; fi; \
	echo "Admin KMS setup complete for: $$MEMBER_LIST"

# ------------------------------------------------------------------------------------ #
#                                        Cleanup                                       #
# ------------------------------------------------------------------------------------ #

clean: tf-secrets-clean
	rm -rf "$(TF_DIR)/.terraform" \
		"$(TF_DIR)/.terraform.lock.hcl" \
		"$(TF_DIR)/terraform.tfstate" \
		"$(TF_DIR)/terraform.tfstate.backup" \
		"$(TF_DIR)/$(TF_PLAN_FILE)"
