# Veritas Nexus Deployment Guide

This guide covers deploying Veritas Nexus in production environments, from single-server deployments to large-scale distributed systems.

## 📋 Table of Contents

1. [Pre-Deployment Checklist](#-pre-deployment-checklist)
2. [Environment Setup](#-environment-setup)
3. [Single Server Deployment](#-single-server-deployment)
4. [Container Deployment](#-container-deployment)
5. [Kubernetes Deployment](#-kubernetes-deployment)
6. [Cloud Deployment](#-cloud-deployment)
7. [Edge Deployment](#-edge-deployment)
8. [Monitoring and Observability](#-monitoring-and-observability)
9. [Security Considerations](#-security-considerations)
10. [Backup and Recovery](#-backup-and-recovery)
11. [Scaling and Load Balancing](#-scaling-and-load-balancing)
12. [Maintenance and Updates](#-maintenance-and-updates)

## ✅ Pre-Deployment Checklist

### Legal and Ethical Requirements

```checklist
- [ ] Legal review completed
- [ ] Ethical guidelines established
- [ ] Compliance requirements identified (GDPR, CCPA, etc.)
- [ ] Data handling policies defined
- [ ] User consent mechanisms implemented
- [ ] Audit trail requirements specified
- [ ] Human oversight processes established
```

### Technical Requirements

```checklist
- [ ] Hardware requirements calculated
- [ ] Network bandwidth assessed
- [ ] Storage requirements estimated
- [ ] Security requirements defined
- [ ] Performance benchmarks established
- [ ] Disaster recovery plan created
- [ ] Monitoring strategy designed
```

### Operational Requirements

```checklist
- [ ] Team training completed
- [ ] Runbooks created
- [ ] Escalation procedures defined
- [ ] Maintenance windows scheduled
- [ ] Support processes established
- [ ] Documentation reviewed
- [ ] Testing completed
```

## 🏗️ Environment Setup

### System Requirements

**Minimum Production Requirements:**
- **CPU**: 8+ cores (Intel Xeon/AMD EPYC recommended)
- **Memory**: 16GB RAM (32GB+ recommended)
- **Storage**: 100GB SSD (NVMe recommended)
- **Network**: 1Gbps bandwidth
- **OS**: Ubuntu 20.04+, RHEL 8+, or similar

**Recommended High-Performance Setup:**
- **CPU**: 16+ cores with AVX-512 support
- **Memory**: 64GB+ RAM
- **GPU**: NVIDIA A100/V100 or RTX 4090 (8GB+ VRAM)
- **Storage**: 500GB+ NVMe SSD
- **Network**: 10Gbps+ bandwidth

### Software Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    libffi-dev \
    python3-dev \
    cmake \
    ninja-build

# Install Rust (latest stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install GPU dependencies (if using CUDA)
# Follow NVIDIA CUDA installation guide for your system
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cuda
```

### Model Downloads

```bash
# Create model directory
sudo mkdir -p /opt/veritas-nexus/models
sudo chown $USER:$USER /opt/veritas-nexus/models

# Download required models
cd /opt/veritas-nexus/models

# Vision models
curl -L https://github.com/yourusername/veritas-nexus/releases/latest/download/vision-models.tar.gz | tar xz

# Audio models
curl -L https://github.com/yourusername/veritas-nexus/releases/latest/download/audio-models.tar.gz | tar xz

# Text models
curl -L https://github.com/yourusername/veritas-nexus/releases/latest/download/text-models.tar.gz | tar xz

# Set permissions
find /opt/veritas-nexus/models -type f -exec chmod 644 {} \;
find /opt/veritas-nexus/models -type d -exec chmod 755 {} \;
```

## 🖥️ Single Server Deployment

### Building for Production

```bash
# Clone repository
git clone https://github.com/yourusername/veritas-nexus.git
cd veritas-nexus

# Build with production optimizations
cargo build --release --features "gpu,parallel,mcp,simd-avx2"

# Install binary
sudo cp target/release/veritas-nexus /usr/local/bin/
sudo chmod +x /usr/local/bin/veritas-nexus

# Create configuration directory
sudo mkdir -p /etc/veritas-nexus
sudo chown $USER:$USER /etc/veritas-nexus
```

### Configuration

Create production configuration file `/etc/veritas-nexus/config.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8080
workers = 8
max_connections = 1000
keepalive_timeout = 75
request_timeout = 30

[tls]
enabled = true
cert_file = "/etc/ssl/certs/veritas-nexus.crt"
key_file = "/etc/ssl/private/veritas-nexus.key"
protocols = ["TLSv1.2", "TLSv1.3"]

[authentication]
enabled = true
method = "api_key"  # or "jwt", "oauth2"
api_keys_file = "/etc/veritas-nexus/api_keys.json"

[rate_limiting]
enabled = true
requests_per_minute = 1000
burst_size = 100
window_size = "1m"

[models]
base_path = "/opt/veritas-nexus/models"
vision_model = "face_detection_v3.onnx"
audio_model = "voice_stress_v2.onnx"
text_model = "roberta_deception_v1.onnx"

[gpu]
enabled = true
device_id = 0
memory_limit_mb = 4096
batch_size = 32
fp16_inference = true

[memory]
pool_size_mb = 1024
cache_size_mb = 512
gc_threshold_mb = 256
enable_memory_mapping = true

[logging]
level = "info"
format = "json"
file = "/var/log/veritas-nexus/app.log"
rotate_size = "100MB"
rotate_count = 10

[metrics]
enabled = true
port = 9090
path = "/metrics"
export_interval = "10s"

[health]
enabled = true
port = 8081
path = "/health"
check_interval = "30s"
```

### Systemd Service

Create service file `/etc/systemd/system/veritas-nexus.service`:

```ini
[Unit]
Description=Veritas Nexus Multi-Modal Lie Detection Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=veritas
Group=veritas
WorkingDirectory=/opt/veritas-nexus
ExecStart=/usr/local/bin/veritas-nexus server --config /etc/veritas-nexus/config.toml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
LimitNOFILE=65536
Environment=RUST_LOG=info
Environment=VERITAS_ENV=production

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/veritas-nexus /tmp
PrivateTmp=true
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictSUIDSGID=true

[Install]
WantedBy=multi-user.target
```

### User and Permissions Setup

```bash
# Create service user
sudo useradd --system --shell /bin/false --home /opt/veritas-nexus veritas

# Create directories
sudo mkdir -p /var/log/veritas-nexus
sudo mkdir -p /var/lib/veritas-nexus
sudo mkdir -p /run/veritas-nexus

# Set ownership
sudo chown -R veritas:veritas /opt/veritas-nexus
sudo chown -R veritas:veritas /var/log/veritas-nexus
sudo chown -R veritas:veritas /var/lib/veritas-nexus
sudo chown -R veritas:veritas /run/veritas-nexus

# Set permissions
sudo chmod 750 /opt/veritas-nexus
sudo chmod 755 /var/log/veritas-nexus
sudo chmod 755 /var/lib/veritas-nexus
sudo chmod 755 /run/veritas-nexus

# Allow veritas user to access GPU (if applicable)
sudo usermod -a -G video veritas

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable veritas-nexus
sudo systemctl start veritas-nexus

# Check status
sudo systemctl status veritas-nexus
```

## 🐳 Container Deployment

### Dockerfile

```dockerfile
# Build stage
FROM rust:1.75-bullseye as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Build application
RUN cargo build --release --features "gpu,parallel,mcp,simd-avx2"

# Runtime stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    libc6 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA runtime (for GPU support)
RUN apt-get update && apt-get install -y \
    libnvidia-compute-470 \
    libnvidia-ml1 \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --system --shell /bin/false --home /app veritas

# Create directories
RUN mkdir -p /app/models /app/config /app/logs /app/data
RUN chown -R veritas:veritas /app

# Copy binary from build stage
COPY --from=builder /app/target/release/veritas-nexus /usr/local/bin/veritas-nexus
RUN chmod +x /usr/local/bin/veritas-nexus

# Copy models and configuration
COPY models/ /app/models/
COPY config/production.toml /app/config/config.toml

# Set ownership
RUN chown -R veritas:veritas /app

# Switch to application user
USER veritas

# Set working directory
WORKDIR /app

# Expose ports
EXPOSE 8080 9090 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Start application
CMD ["veritas-nexus", "server", "--config", "/app/config/config.toml"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  veritas-nexus:
    build: .
    container_name: veritas-nexus
    restart: unless-stopped
    ports:
      - "8080:8080"   # API port
      - "9090:9090"   # Metrics port
      - "8081:8081"   # Health check port
    volumes:
      - ./models:/app/models:ro
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./data:/app/data
    environment:
      - RUST_LOG=info
      - VERITAS_ENV=production
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - veritas-network
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    container_name: veritas-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    networks:
      - veritas-network

  postgres:
    image: postgres:15-alpine
    container_name: veritas-postgres
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=veritas
      - POSTGRES_USER=veritas
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - veritas-network

  nginx:
    image: nginx:alpine
    container_name: veritas-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - veritas-nexus
    networks:
      - veritas-network

  prometheus:
    image: prom/prometheus:latest
    container_name: veritas-prometheus
    restart: unless-stopped
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - veritas-network

  grafana:
    image: grafana/grafana:latest
    container_name: veritas-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    networks:
      - veritas-network

volumes:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:

networks:
  veritas-network:
    driver: bridge
```

### Build and Deploy

```bash
# Create environment file
cat > .env << EOF
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=your_secure_password
GRAFANA_PASSWORD=your_secure_password
EOF

# Build and start services
docker-compose up -d

# Check logs
docker-compose logs -f veritas-nexus

# Scale services
docker-compose up -d --scale veritas-nexus=3
```

## ⚓ Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: veritas-nexus
---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: veritas-config
  namespace: veritas-nexus
data:
  config.toml: |
    [server]
    host = "0.0.0.0"
    port = 8080
    workers = 8
    
    [models]
    base_path = "/app/models"
    
    [gpu]
    enabled = true
    device_id = 0
    memory_limit_mb = 4096
    
    [logging]
    level = "info"
    format = "json"
    
    [metrics]
    enabled = true
    port = 9090
    path = "/metrics"
```

### Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: veritas-secrets
  namespace: veritas-nexus
type: Opaque
data:
  api-key: <base64-encoded-api-key>
  tls-cert: <base64-encoded-tls-cert>
  tls-key: <base64-encoded-tls-key>
  db-password: <base64-encoded-db-password>
```

### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: veritas-nexus
  namespace: veritas-nexus
  labels:
    app: veritas-nexus
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: veritas-nexus
  template:
    metadata:
      labels:
        app: veritas-nexus
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: veritas-nexus
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      initContainers:
      - name: model-downloader
        image: busybox:1.35
        command:
        - sh
        - -c
        - |
          if [ ! -f /models/vision-model.onnx ]; then
            echo "Downloading models..."
            wget -q -O /tmp/models.tar.gz https://github.com/yourusername/veritas-nexus/releases/latest/download/models.tar.gz
            tar -xzf /tmp/models.tar.gz -C /models/
          fi
        volumeMounts:
        - name: models-volume
          mountPath: /models
      containers:
      - name: veritas-nexus
        image: veritas-nexus:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP
        - containerPort: 9090
          name: metrics
          protocol: TCP
        - containerPort: 8081
          name: health
          protocol: TCP
        env:
        - name: RUST_LOG
          value: "info"
        - name: VERITAS_ENV
          value: "production"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: models-volume
          mountPath: /app/models
          readOnly: true
        - name: secrets-volume
          mountPath: /app/secrets
          readOnly: true
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30
      volumes:
      - name: config-volume
        configMap:
          name: veritas-config
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
      - name: secrets-volume
        secret:
          secretName: veritas-secrets
      nodeSelector:
        accelerator: nvidia-tesla-v100  # GPU node selector
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

### Services and Ingress

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: veritas-nexus-service
  namespace: veritas-nexus
  labels:
    app: veritas-nexus
spec:
  type: ClusterIP
  ports:
  - port: 8080
    targetPort: 8080
    protocol: TCP
    name: http
  - port: 9090
    targetPort: 9090
    protocol: TCP
    name: metrics
  selector:
    app: veritas-nexus
---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: veritas-nexus-ingress
  namespace: veritas-nexus
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.veritas-nexus.example.com
    secretName: veritas-tls
  rules:
  - host: api.veritas-nexus.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: veritas-nexus-service
            port:
              number: 8080
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: veritas-nexus-hpa
  namespace: veritas-nexus
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: veritas-nexus
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
```

### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Check deployment status
kubectl get pods -n veritas-nexus
kubectl get services -n veritas-nexus
kubectl get ingress -n veritas-nexus

# View logs
kubectl logs -f deployment/veritas-nexus -n veritas-nexus

# Scale deployment
kubectl scale deployment veritas-nexus --replicas=5 -n veritas-nexus
```

## ☁️ Cloud Deployment

### AWS EKS

```bash
# Create EKS cluster
eksctl create cluster \
  --name veritas-nexus \
  --region us-west-2 \
  --node-type p3.2xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --with-oidc \
  --ssh-access \
  --ssh-public-key my-key \
  --managed

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# Install AWS Load Balancer Controller
eksctl utils associate-iam-oidc-provider --region=us-west-2 --cluster=veritas-nexus --approve
eksctl create iamserviceaccount \
  --cluster=veritas-nexus \
  --namespace=kube-system \
  --name=aws-load-balancer-controller \
  --attach-policy-arn=arn:aws:iam::123456789012:policy/AWSLoadBalancerControllerIAMPolicy \
  --override-existing-serviceaccounts \
  --approve

# Install Cluster Autoscaler
eksctl create iamserviceaccount \
  --cluster=veritas-nexus \
  --namespace=kube-system \
  --name=cluster-autoscaler \
  --attach-policy-arn=arn:aws:iam::123456789012:policy/ClusterAutoscalerPolicy \
  --override-existing-serviceaccounts \
  --approve
```

### GCP GKE

```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create veritas-nexus \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --num-nodes=3 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10 \
  --enable-autorepair \
  --enable-autoupgrade

# Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster=veritas-nexus \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --num-nodes=2 \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=5

# Install NVIDIA drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### Azure AKS

```bash
# Create AKS cluster
az aks create \
  --resource-group veritas-nexus-rg \
  --name veritas-nexus \
  --location eastus \
  --node-count 3 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10 \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group veritas-nexus-rg --name veritas-nexus

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/Azure/azhpc-extensions/master/NvidiaGPU/resources.yaml
```

## 🌐 Edge Deployment

### Edge Optimized Configuration

```toml
# edge-config.toml
[server]
host = "0.0.0.0"
port = 8080
workers = 2  # Reduced for edge devices

[models]
base_path = "/opt/veritas-nexus/models"
# Use quantized models for edge deployment
vision_model = "face_detection_quantized_int8.onnx"
audio_model = "voice_stress_quantized_int8.onnx"
text_model = "distilbert_quantized_int8.onnx"

[gpu]
enabled = false  # Often not available on edge
memory_limit_mb = 512

[memory]
pool_size_mb = 256  # Reduced memory usage
cache_size_mb = 64
gc_threshold_mb = 32

[performance]
# Edge-specific optimizations
enable_model_quantization = true
enable_pruning = true
max_batch_size = 4
enable_streaming = false  # Reduce memory usage

[offline]
# Edge devices often work offline
enable_offline_mode = true
cache_models_locally = true
enable_model_fallback = true
```

### NVIDIA Jetson Deployment

```bash
# Install JetPack (on Jetson device)
sudo apt update
sudo apt install nvidia-jetpack

# Install Rust for ARM64
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup target add aarch64-unknown-linux-gnu

# Build for Jetson
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
cargo build --release --target aarch64-unknown-linux-gnu \
  --features "parallel,simd-neon"

# Deploy to Jetson
scp target/aarch64-unknown-linux-gnu/release/veritas-nexus jetson@192.168.1.100:/home/jetson/
scp edge-config.toml jetson@192.168.1.100:/home/jetson/config.toml

# Setup service on Jetson
ssh jetson@192.168.1.100 'sudo mv veritas-nexus /usr/local/bin/ && sudo chmod +x /usr/local/bin/veritas-nexus'
```

### Intel NUC/Edge Server

```dockerfile
# Dockerfile.edge
FROM rust:1.75 as builder

WORKDIR /app
COPY . .

# Build with Intel optimizations
RUN cargo build --release --features "parallel,simd-avx2" \
    --target x86_64-unknown-linux-gnu

FROM ubuntu:22.04

# Install Intel OpenVINO runtime
RUN apt-get update && apt-get install -y \
    intel-openvino-runtime-cpu \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/x86_64-unknown-linux-gnu/release/veritas-nexus /usr/local/bin/
COPY edge-config.toml /app/config.toml

EXPOSE 8080

CMD ["veritas-nexus", "server", "--config", "/app/config.toml"]
```

## 📊 Monitoring and Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "veritas_rules.yml"

scrape_configs:
  - job_name: 'veritas-nexus'
    static_configs:
      - targets: ['veritas-nexus:9090']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'veritas-health'
    static_configs:
      - targets: ['veritas-nexus:8081']
    metrics_path: /health
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "title": "Veritas Nexus Monitoring",
    "tags": ["veritas", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(veritas_requests_total[5m])",
            "legendFormat": "{{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(veritas_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(veritas_requests_total{status!~\"2..\"}[5m]) / rate(veritas_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "veritas_gpu_utilization_percent",
            "legendFormat": "GPU {{device}}"
          }
        ]
      }
    ]
  }
}
```

### Log Aggregation with ELK Stack

```yaml
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "veritas-nexus" {
    json {
      source => "message"
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if [level] == "ERROR" {
      mutate {
        add_tag => [ "error" ]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "veritas-nexus-%{+YYYY.MM.dd}"
  }
}
```

## 🔒 Security Considerations

### Network Security

```bash
# UFW firewall rules
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8080/tcp  # API (internal only)
sudo ufw enable

# Fail2ban configuration
sudo apt install fail2ban
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 3
EOF
```

### TLS Configuration

```nginx
# nginx SSL configuration
server {
    listen 443 ssl http2;
    server_name api.veritas-nexus.example.com;

    ssl_certificate /etc/ssl/certs/veritas-nexus.crt;
    ssl_certificate_key /etc/ssl/private/veritas-nexus.key;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### API Key Management

```rust
// Generate secure API keys
use rand::RngCore;
use base64::{Engine as _, engine::general_purpose};

fn generate_api_key() -> String {
    let mut key = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut key);
    general_purpose::URL_SAFE_NO_PAD.encode(key)
}

// Store in secure configuration
let api_keys = vec![
    ("client_1", generate_api_key()),
    ("client_2", generate_api_key()),
    ("admin", generate_api_key()),
];
```

## 💾 Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/opt/backups/veritas-nexus"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="veritas"

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
pg_dump -h localhost -U veritas $DB_NAME | gzip > $BACKUP_DIR/db_backup_$DATE.sql.gz

# Model backup
tar -czf $BACKUP_DIR/models_backup_$DATE.tar.gz /opt/veritas-nexus/models/

# Configuration backup
tar -czf $BACKUP_DIR/config_backup_$DATE.tar.gz /etc/veritas-nexus/

# Log backup
tar -czf $BACKUP_DIR/logs_backup_$DATE.tar.gz /var/log/veritas-nexus/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

### Disaster Recovery Plan

```yaml
# disaster-recovery.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: disaster-recovery-plan
data:
  plan.md: |
    # Disaster Recovery Plan
    
    ## Recovery Time Objectives (RTO)
    - Critical services: 15 minutes
    - Full system: 2 hours
    
    ## Recovery Point Objectives (RPO)
    - Database: 15 minutes
    - Models: 24 hours
    - Configuration: 1 hour
    
    ## Recovery Procedures
    
    ### Database Recovery
    1. Restore from latest backup
    2. Apply transaction logs
    3. Verify data integrity
    
    ### Service Recovery
    1. Deploy from latest container image
    2. Restore configuration
    3. Verify health checks
    
    ### Model Recovery
    1. Download from model repository
    2. Verify checksums
    3. Test inference
```

## 📈 Scaling and Load Balancing

### Horizontal Scaling

```yaml
# horizontal-scaling.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: veritas-nexus-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: veritas-nexus
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: veritas_requests_per_second
      target:
        type: AverageValue
        averageValue: "50"
```

### Load Balancer Configuration

```nginx
# load-balancer.conf
upstream veritas_backend {
    least_conn;
    server 10.0.1.10:8080 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8080 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8080 weight=1 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    server_name api.veritas-nexus.example.com;
    
    location / {
        proxy_pass http://veritas_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://veritas_backend/health;
    }
}
```

## 🔧 Maintenance and Updates

### Rolling Updates

```bash
#!/bin/bash
# rolling-update.sh

NEW_VERSION=$1
if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <new_version>"
    exit 1
fi

echo "Starting rolling update to version $NEW_VERSION"

# Update Kubernetes deployment
kubectl set image deployment/veritas-nexus \
    veritas-nexus=veritas-nexus:$NEW_VERSION \
    -n veritas-nexus

# Wait for rollout to complete
kubectl rollout status deployment/veritas-nexus -n veritas-nexus

# Verify deployment
kubectl get pods -n veritas-nexus
kubectl logs -f deployment/veritas-nexus -n veritas-nexus --tail=10

echo "Rolling update completed"
```

### Blue-Green Deployment

```yaml
# blue-green-deployment.yaml
apiVersion: v1
kind: Service
metadata:
  name: veritas-nexus-active
  namespace: veritas-nexus
spec:
  selector:
    app: veritas-nexus
    version: blue  # Switch between blue/green
  ports:
  - port: 8080
    targetPort: 8080

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: veritas-nexus-blue
  namespace: veritas-nexus
spec:
  replicas: 3
  selector:
    matchLabels:
      app: veritas-nexus
      version: blue
  template:
    metadata:
      labels:
        app: veritas-nexus
        version: blue
    spec:
      containers:
      - name: veritas-nexus
        image: veritas-nexus:v1.0.0
        # ... container spec
```

### Health Checks and Monitoring

```bash
#!/bin/bash
# health-check.sh

ENDPOINT="https://api.veritas-nexus.example.com/health"
TIMEOUT=10
MAX_RETRIES=3

for i in $(seq 1 $MAX_RETRIES); do
    if curl -f -s --max-time $TIMEOUT $ENDPOINT > /dev/null; then
        echo "Health check passed"
        exit 0
    else
        echo "Health check failed (attempt $i/$MAX_RETRIES)"
        sleep 5
    fi
done

echo "Health check failed after $MAX_RETRIES attempts"
exit 1
```

---

This deployment guide provides comprehensive coverage of deploying Veritas Nexus in various environments. Choose the deployment method that best fits your infrastructure and requirements. For additional support, consult the [Troubleshooting Guide](TROUBLESHOOTING.md) or contact our support team.