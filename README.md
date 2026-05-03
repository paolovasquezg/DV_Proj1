# Data Visualization: Project 1

An attempted replica of [VAST Challenge 2019 MC1](https://github.com/na399/VAST-Challenge-2019-MC1) — an interactive dashboard for exploring seismic and damage data from the 2019 VAST Challenge Mini-Challenge 1.

## How to run

### 1. Transform the data

The `transformations/` directory contains a Python pipeline that preprocesses the raw data into the format consumed by the dashboard.

```bash
cd transformations
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python transform.py
```

This will produce the transformed data files needed by the dashboard.

### 2. Run the dashboard

```bash
cd dashboard
npm install
npm run dev
```

Open the local URL: `http://localhost:5173`.
