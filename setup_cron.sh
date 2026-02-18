#!/usr/bin/env bash
# =============================================================================
# Setup cron job for daily options analysis at 7:30 AM (Mon-Fri)
# Usage: bash setup_cron.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_daily.sh"
CRON_SCHEDULE="30 7 * * 1-5"  # 7:30 AM, Monday through Friday
CRON_JOB="${CRON_SCHEDULE} ${RUNNER}"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -qF "${RUNNER}"; then
    echo "[INFO] Cron job already exists:"
    crontab -l | grep -F "${RUNNER}"
    echo ""
    echo "To remove it: crontab -e  (and delete the line)"
else
    # Add the cron job
    (crontab -l 2>/dev/null || true; echo "${CRON_JOB}") | crontab -
    echo "[OK] Cron job installed:"
    echo "  ${CRON_JOB}"
    echo ""
    echo "The options analyzer will run every weekday at 7:30 AM."
    echo "Reports will be saved to: ${SCRIPT_DIR}/reports/"
    echo "Logs will be saved to: ${SCRIPT_DIR}/logs/"
fi

echo ""
echo "Current crontab:"
crontab -l 2>/dev/null || echo "(empty)"
