# CockpitSentinel — Cloud & Edge Deployment Architecture Guide

This guide explains how to deploy **CockpitSentinel** to the cloud (AWS and modern cloud providers), how the **telemetry dashboard and live video field** interact between the in-cabin vehicle and the cloud, and how to set up an automated **CI/CD update pipeline** for zero-downtime releases.

---

## 1. High-Level System Architecture: Edge vs. Cloud

In automotive Driver Monitoring Systems (DMS) and fleet telematics, **safety-critical AI processing must NEVER rely solely on a remote cloud connection**. 

```
┌─────────────────────────────────────────────────────────┐
│               IN-VEHICLE EDGE (Physical Car)            │
│  - In-Cabin NIR/RGB Camera (/dev/video0)                │
│  - Edge Compute (Jetson Orin Nano, RPi 5, or PC/Laptop) │
│  - Real-Time MediaPipe (EAR/MAR/Yaw) + YOLO11s (30+ FPS)│
│  - Local Instant In-Cabin Audio Alarm / Buzzer (<30ms)  │
└────────────────────────────┬────────────────────────────┘
                             │ 4G / LTE / 5G / Wi-Fi
                             │ (HTTPS REST API / WebRTC)
                             ▼
┌─────────────────────────────────────────────────────────┐
│               AWS CLOUD / MODERN PAAS                   │
│  - Dockerized FastAPI Service (Port 8000)               │
│  - Fleet Telemetry Dashboard & Real-Time Analytics UI   │
│  - Driver Biometric Profiles & Threshold Calibrator     │
│  - Commercial Route Management & Waypoint Checkpoints   │
│  - Driver Safety Leaderboard & Alert Incident Audits    │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                 CENTRALIZED DATABASE                    │
│  - MongoDB Atlas (Cloud Cluster M0 Free or Dedicated)   │
│  - SQLite Fallback (for zero-connectivity resilience)   │
└─────────────────────────────────────────────────────────┘
```

---

## 2. How the Dashboard & Live Field Work

### A. The Telemetry Dashboard (`app.py`)
- **Execution**: Runs inside a lightweight Docker container on AWS (App Runner, ECS, or EC2).
- **Database**: Binds to **MongoDB Atlas** using the `MONGODB_URI` environment variable.
- **Access**: Fleet dispatchers and safety managers can log in from anywhere in the world on desktop, tablet, or smartphone to:
  - Monitor fleet attention index % and safety ratings in real time.
  - Dynamically calibrate per-driver sensitivity thresholds (EAR, MAR, Head Yaw, Phone/Smoking confidence).
  - Inspect audit trails of distraction incidents with timestamps and risk scores.
  - Manage routes, commercial transit corridors, and driver assignments.

### B. The Live Field (Camera & AI Inference)
Because the AWS cloud data center is hundreds of kilometers away from the physical driver, there are **two operational modes** for the live field:

#### Mode 1: Production Automotive DMS (Edge AI + Cloud Telemetry) — *Industry Standard*
1. **Local Inference**: The in-cabin edge device runs `cockpit-sentinel` locally against the physical camera.
2. **Instant Local Alarms**: When eye closure ($EAR < 0.22$) or phone usage is sustained for $>2.0s$, the in-cabin alarm sounds **immediately (<30ms)**. This works even when the car is inside an underground parking garage or remote highway with **zero cellular reception**.
3. **Cloud Telemetry Uplink**: As driving proceeds, the edge device transmits compact telemetry packets to the AWS dashboard:
   - `POST /api/trips`: Duration, attention %, average FPS, safety star rating.
   - `POST /api/alerts`: Incident timestamps, active signals, risk level, and duration.
4. **Remote Snapshot Feeds**: On `CRITICAL` or `WARNING` alerts, the edge device uploads a compressed 640×640 snapshot frame or 3-second animated clip so fleet managers can verify the incident directly from the dashboard.

#### Mode 2: Remote Live Stream / Web Demo Mode
If you want to view a real-time live video feed directly on the AWS web dashboard:
1. **Browser Webcam Demo**: The client browser accesses the local webcam via HTML5 `navigator.mediaDevices.getUserMedia()`, sends frames over WebSocket/WebRTC to the FastAPI service, and receives annotated frames back with HUD bounding boxes.
2. **In-Vehicle RTSP / WebRTC Stream**: The vehicle edge device streams a low-bandwidth H.264 stream to an AWS media gateway (such as **MediaMTX** or **AWS Kinesis Video Streams**). The dashboard embeds a standard video player that activates only when a dispatcher requests live view, saving cellular bandwidth.

---

## 3. Best Deployment Strategy on AWS

### Option A: AWS App Runner (Recommended — Simplest, Serverless & Low Maintenance)
**AWS App Runner** is a fully managed container service that automatically builds, deploys, scales, and provisions SSL certificates without configuring VPCs, load balancers, or EC2 instances.

#### Step 1: Push Container to Amazon ECR
```bash
# Authenticate Docker to AWS ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.ap-south-1.amazonaws.com

# Create an ECR repository
aws ecr create-repository --repository-name cockpit-sentinel --region ap-south-1

# Build and tag your image
docker build -t cockpit-sentinel:latest .
docker tag cockpit-sentinel:latest <aws_account_id>.dkr.ecr.ap-south-1.amazonaws.com/cockpit-sentinel:latest

# Push image
docker push <aws_account_id>.dkr.ecr.ap-south-1.amazonaws.com/cockpit-sentinel:latest
```

#### Step 2: Create App Runner Service
1. In the **AWS App Runner Console**, choose **Container registry** > **Amazon ECR**.
2. Select your repository: `cockpit-sentinel:latest`.
3. Deployment trigger: **Automatic** (deploys every time a new image is pushed).
4. Service settings:
   - Port: `8000`
   - CPU: `1 vCPU`, Memory: `2 GB`
   - Environment Variables:
     - `DATABASE_BACKEND`: `mongodb`
     - `MONGODB_URI`: `mongodb+srv://<user>:<password>@cluster0.mongodb.net/cockpit_sentinel?retryWrites=true&w=majority`
     - `DASHBOARD_PORT`: `8000`
5. Click **Create & Deploy**. In ~3 minutes, AWS provisions an HTTPS URL (e.g. `https://xxxxxx.ap-south-1.awsapprunner.com`).

---

### Option B: AWS EC2 (t4g.small / t3.medium) with Docker Compose (Best for Full Control & Fixed Cost)
If you prefer a fixed monthly cost ($8–$15/month) with full SSH control:

1. Launch an **Ubuntu 24.04 LTS** EC2 instance (`t4g.small` ARM or `t3.medium` x86).
2. Install Docker & Docker Compose:
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose-v2
   sudo usermod -aG docker ubuntu
   ```
3. Clone repository and set up environment:
   ```bash
   git clone https://github.com/<your-username>/CockpitSentinel.git
   cd CockpitSentinel
   cp .env.example .env
   nano .env  # set MONGODB_URI and DASHBOARD_PORT=8000
   ```
4. Start the service with Docker Compose:
   ```bash
   docker compose up -d --build
   ```
5. Install **Caddy** for instant, automatic free Let's Encrypt SSL:
   ```bash
   sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
   sudo apt update && sudo apt install caddy
   ```
   Add to `/etc/caddy/Caddyfile`:
   ```caddy
   cockpit.yourdomain.com {
       reverse_proxy localhost:8000
   }
   ```
   Reload Caddy: `sudo systemctl reload caddy`.

---

## 4. Best Way to Update Code (Automated CI/CD Pipeline)

The industry best practice is **Git-Driven Automation** using GitHub Actions:

```
Developer Push (git push origin main)
         │
         ▼
GitHub Actions CI
  ├── 1. Run Ruff linting & Pytest unit tests
  ├── 2. Build Docker container image
  ├── 3. Authenticate to Amazon ECR
  └── 4. Push image tagged with commit SHA & 'latest'
         │
         ▼
AWS App Runner / EC2 Watchtower
  └── Automatically pulls new image and performs
      Zero-Downtime Rolling Update (Health check at /api/status)
```

### GitHub Actions Deployment Workflow (`.github/workflows/deploy.yml`)
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Run Tests
        run: |
          pip install -r requirements.txt pytest
          pytest tests/unit/

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-south-1

      - name: Log in to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and Push Docker Image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: cockpit-sentinel
          IMAGE_TAG: latest
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

      - name: Trigger App Runner Deployment
        run: |
          aws apprunner start-deployment \
            --service-arn ${{ secrets.APPRUNNER_SERVICE_ARN }} \
            --region ap-south-1
```

### Managing Large Model Weights Updates
- **Do not commit `.pt` or `.engine` weights to Git.**
- Store trained weights in an **AWS S3 bucket**: `s3://cockpit-sentinel-models/distraction/best.pt`.
- Use a startup script or versioned download in Docker or Edge devices to pull weights upon deployment:
  ```bash
  aws s3 sync s3://cockpit-sentinel-models/ ./weights/trained/
  ```

---

## 5. Provider Comparison Matrix

| Cloud Provider | Platform Option | Monthly Cost | Ease of Setup | Maintenance Overhead | Best Suited For |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **AWS** | **App Runner** | $5 – $20 | ★★★★★ | **Zero Ops** (Serverless) | Production fleet teams already using AWS. |
| **AWS** | **EC2 + Docker** | $8 – $15 | ★★★☆☆ | Low (Manage Linux updates) | Budget AWS deployment with full terminal SSH. |
| **Railway / Render** | **Web Service** | $5 (Free Tier) | ★★★★★ | **Zero Ops** (Push to Git) | Easiest & fastest MVP deployment (<5 mins). |
| **DigitalOcean** | **App Platform** | $5 – $12 | ★★★★☆ | Low (Managed PaaS) | Clean UI, simple billing, predictable pricing. |

### Final Verdict:
1. **If you strictly need AWS**: Choose **AWS App Runner + Amazon ECR + MongoDB Atlas**. It is modern, auto-scaling, requires zero server administration, and deploys automatically on every `git push`.
2. **If you want the simplest, fastest deployment anywhere**: Choose **Railway** or **Render** connected to your GitHub repository. It deploys the Docker container automatically in under 3 minutes with free SSL.
3. **For the in-cabin vehicle live field**: Keep the AI inference running at the **Edge** (Jetson, Raspberry Pi, or Laptop) and let it send telemetry and alert events upstream to your cloud dashboard via HTTPS.
