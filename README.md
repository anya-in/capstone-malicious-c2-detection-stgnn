# Resilient ST-GNNs for Malicious C2 Detection

## Problem Overview
Modern malware has reached a "Total Eclipse" of network visibility using **Encrypted Client Hello (ECH)**. Our team observed that 2026-era threats use Adversarial AI to hide heartbeats as benign traffic. We are building an ST-GNN framework to identify "Fingerprints of Intent" without needing to decrypt data.

## SMART Goals
- **Specific**: Targeting "Total Eclipse" scenarios where SNI/Certificates are hidden.
- **Measurable**: Aiming for **>98% Precision** and **<10ms latency**.
- **Assignable**: Developed by Lavanya and Shivani (Network Engineering & Deep Learning focus).
- **Realistic**: Utilizing INT8 quantization and ST-GNN architectures.
- **Time-related**: Final delivery by May 5, 2026.

## Getting Started
1. **Clone the Repo**: `git clone https://github.com/[username]/ech-resilient-stgnn.git`
2. **Setup Env**: `pip install -r requirements.txt`
3. **Run Blinding Script**: `python src/preprocessing/blind_data.py`

## Notion Roadmap & Research
We maintain a centralized **Notion workspace** where we track our 25-paper literature review, individual task ownership, and daily progress logs.
