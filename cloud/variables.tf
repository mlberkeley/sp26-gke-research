variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "Default GCP region."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Default GCP zone."
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "Name of the GKE cluster."
  type        = string
  default     = "research-agent"
}

variable "network_name" {
  description = "VPC network name."
  type        = string
  default     = "research-vpc"
}

variable "subnetwork_name" {
  description = "Subnet name."
  type        = string
  default     = "research-subnet"
}

variable "subnetwork_cidr" {
  description = "CIDR range for the subnet."
  type        = string
  default     = "10.10.0.0/16"
}

variable "node_count" {
  description = "Number of nodes in the primary node pool."
  type        = number
  default     = 1
}

variable "node_machine_type" {
  description = "Machine type for GKE nodes."
  type        = string
  default     = "e2-standard-2"
}

variable "node_disk_type" {
  description = "Boot disk type for GKE nodes. Use pd-standard to avoid SSD quota usage."
  type        = string
  default     = "pd-standard"
}

variable "node_disk_size_gb" {
  description = "Boot disk size in GB for GKE nodes."
  type        = number
  default     = 30
}

variable "gcp_credentials" {
  description = "Optional service account JSON content. Leave empty to use ADC from gcloud login."
  type        = string
  sensitive   = true
  default     = ""
}

variable "dummy_secret" {
  description = "Placeholder secret to keep SOPS workflow active when no real secret is needed."
  type        = string
  sensitive   = true
  default     = ""
}
