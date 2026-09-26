# AdaptArena

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/McGill-NLP/web-agents-test-time-adaptation.git
cd web-agents-test-time-adaptation
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
playwright install
```

---

## Environment Variables

Set your API key:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

---

## Configuration

```bash
CONFIG_NAME="deployment_task.json"
SUFFIX="ds-3"

export SUFFIX=$SUFFIX
export WEBSYNTH_CONFIG_NAME=$CONFIG_NAME

source vars/set_envs.sh
```

---

## Run

```bash
python run_agent.py \
  --benchmark websynth \
  --model gpt-5-mini
```

---

## 📝 Notes

- Ensure all environment variables are set before running.
- The task file (`deployment_task.json`) controls experiment settings.
