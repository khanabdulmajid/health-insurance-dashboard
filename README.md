# Health Insurance Dashboard

## Setup
```bash
git clone https://github.com/khanabdulmajid/health-insurance-dashboard.git
cd health-insurance-dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py runserver
