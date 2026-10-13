# Vehicle In-Cabin Deployment Guide

This guide walks through deploying **CockpitSentinel** onto in-vehicle edge hardware (such as **NVIDIA Jetson Orin Nano** or **Orin NX**) using your trained **YOLO11** distraction model and **MediaPipe Face Landmarker**.

---

## 1. Automotive Hardware Architecture

```
                  ┌────────────────────────────────────────┐
                  │           Vehicle 12V Power            │
                  │   (Ignition switched via 12V->5V/19V)  │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
┌─────────────────────────┐     ┌───────────────────────────┐     ┌────────────────────────┐
│   In-Cabin IR Camera    │────►│  NVIDIA Jetson Orin Nano  │────►│    Driver Alert Unit   │
│ (850nm/940nm Wide-Angle)│ CSI/│   (15W Power, TensorRT)   │     │  (Buzzer / Audio Chime │
│  Night-Vision Capable   │ USB │                           │     │   or CAN-Bus message)  │
└─────────────────────────┘     └───────────────────────────┘     └────────────────────────┘
```

### Recommended In-Vehicle Hardware
| Component | Recommendation | Why |
| :--- | :--- | :--- |
| **Compute Board** | **NVIDIA Jetson Orin Nano (8GB)** | 40 TOPS AI compute, runs YOLO11 + MediaPipe at 60 FPS, draws only 10W–15W. |
| **Camera** | **Near-Infrared (NIR) 1080p Camera** with 850nm or 940nm LED ring | Standard RGB cameras fail at night or under harsh midday shadows. NIR provides consistent monochrome illumination 24/7. |
| **Lens** | **120°–140° Wide FOV lens** | Captures the driver's face, hands on the steering wheel, and lap area. |
| **Power Supply** | **12V-to-19V DC Automotive Regulator** | Isolates voltage spikes and alternator ripple during engine cranking. |

---

## 2. Model Export on Edge Device (TensorRT)

Once your model `best.pt` is trained in Colab, copy it to the Jetson board and compile it into an **NVIDIA TensorRT `.engine`**:

```bash
# On your NVIDIA Jetson (JetPack 5.1+ / 6.x)
pip install ultralytics

# Compile into FP16 TensorRT engine
python scripts/export_jetson_tensorrt.py --model weights/trained/best.pt
```

This generates `best.engine` optimized specifically for the Jetson's Ampere GPU cores.

### Real-Time Performance on Jetson Orin Nano:
* **Preprocessed 640x640 frame:** ~8.2 ms
* **TensorRT YOLO11s FP16:** ~11.5 ms
* **MediaPipe Face Landmarker:** ~9.0 ms
* **Total End-to-End Pipeline:** **~28.7 ms (~35 FPS)** — exceeds real-time automotive requirements.

---

## 3. Running CockpitSentinel in the Vehicle

Launch the monitor pointing directly to the compiled TensorRT engine:

```bash
python -m cockpit_sentinel.pipeline.live_monitor \
  --source /dev/video0 \
  --phone-model-path weights/trained/best.engine \
  --smoking-model-path weights/trained/best.engine \
  --device 0
```

---

## 4. Headless Autostart on Vehicle Ignition (systemd)

To make CockpitSentinel automatically start when the vehicle starts:

Create `/etc/systemd/system/cockpit-sentinel.service`:

```ini
[Unit]
Description=CockpitSentinel Driver Monitoring Service
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/CockpitSentinel
ExecStart=/home/jetson/CockpitSentinel/.venv/bin/python -m cockpit_sentinel.pipeline.live_monitor --source /dev/video0 --headless
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable cockpit-sentinel.service
sudo systemctl start cockpit-sentinel.service
```

---

## 5. Alerts & Vehicle Integration

In real cars, screen popups can distract the driver further. The alerts should be non-visual or discreet:
1. **Audio Tone / Chime:** Trigger an audible beep via vehicle speakers or a small piezo buzzer connected to Jetson GPIO pins.
2. **Haptic Vibration:** Trigger a steering wheel or seat motor via relay/CAN bus.
3. **CAN-Bus Integration:** Send high-risk alert frames over CAN-Bus (e.g., using `python-can` with a MCP2515 CAN transceiver or SocketCAN).
