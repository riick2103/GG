#!/usr/bin/env bash
# =============================================================================
# Daily Options Analyzer Runner
# Schedule this via cron: 30 7 * * 1-5 /path/to/run_daily.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

echo "=== Options Analyzer Run: $(date) ===" | tee "${LOG_FILE}"

# Activate virtual environment
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "[ERROR] Virtual environment not found at ${VENV_DIR}" | tee -a "${LOG_FILE}"
    echo "Run: python3 -m venv ${VENV_DIR} && source ${VENV_DIR}/bin/activate && pip install yfinance pandas tabulate" | tee -a "${LOG_FILE}"
    exit 1
fi

# Run the analyzer
python3 "${SCRIPT_DIR}/market_options_analyzer.py" 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "=== Completed: $(date) ===" | tee -a "${LOG_FILE}"

# Cleanup old logs (keep last 30 days)
find "${LOG_DIR}" -name "run_*.log" -mtime +30 -delete 2>/dev/null || true
find "${SCRIPT_DIR}/reports" -name "options_report_*.txt" -mtime +30 -delete 2>/dev/null || true
