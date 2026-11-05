#!/bin/bash
# trap "sudo pkill -9 -f tcpdump; sudo pkill -9 -f orchestrator" EXIT INT

cleanup() {
    echo ""
    echo "=== Interrupted by user, cleaning up... ==="
    
    # clean orchestrator process
    sudo pkill -9 -f orchestrator 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "[Cleanup] orchestrator processes killed"
    fi
    
    # clean tcpdump process
    sudo pkill -9 -f tcpdump 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "[Cleanup] tcpdump processes killed"
    fi
    
    # clean lo netem 
    sudo tc qdisc del dev lo root netem 2>/dev/null
    sudo tc qdisc del dev lo ingress 2>/dev/null
    echo "[Cleanup] tc rules cleared"
    
    echo "=== Cleanup completed, exiting ==="
    exit 1
}
# catch Ctrl+C (SIGINT) and (SIGTERM)
trap cleanup SIGINT SIGTERM


SCENARIO="chat_token" # realtime_video_understanding, chat_token
RUNS=10
PROFILE_INTERVAL=20

mkdir -p results/reports
mkdir -p results/captures/chat_token

# ============== clean old results ==============
cleanup_profile_files() {
    local profile=$1
    local capture_dir="results/captures/chat_token/${profile}"
    local report_file="results/reports/experiment_report_${SCENARIO}_${profile}.json"
    local db_file="logs/traffic_logs__${SCENARIO}_${profile}.db"
    
    echo "[Cleanup] Removing old files for profile: ${profile}"
    
    #  capture-dir
    if [ -d "$capture_dir" ]; then
        rm -rf "$capture_dir"
        echo "  - Deleted: $capture_dir"
    fi
    
    #  report 文件
    if [ -f "$report_file" ]; then
        rm -f "$report_file"
        echo "  - Deleted: $report_file"
    fi
    
    #  db 文件
    if [ -f "$db_file" ]; then
        rm -f "$db_file"
        echo "  - Deleted: $db_file"
    fi

    #  capture 
    mkdir -p "$capture_dir"
}
# ====================================


for PROFILE in  congested no_emulation lossy 6g_itu_hrllc 5g_urban cell_edge 5qi_7 5qi_80; do
    echo "=== Running: $SCENARIO - $PROFILE ==="

    cleanup_profile_files "$PROFILE"
    echo "[Test] Starting orchestrator..."
    python orchestrator.py \
        --scenario $SCENARIO \
        --profile $PROFILE \
        --runs $RUNS \
        --interface lo \
        --capture-pcap \
        --capture-dir results/captures/chat_token/${PROFILE} \
        --report results/reports/experiment_report_${SCENARIO}_${PROFILE}.json \
        --db logs/traffic_logs__${SCENARIO}_${PROFILE}.db

# For realtime_video_understanding
#    python orchestrator.py \
#        --scenario $SCENARIO \
#        --profile $PROFILE \
#        --runs $RUNS \
#        --interface lo \
#        --capture-pcap \
#        --capture-dir capture/captures/vlm/${PROFILE} \
#        --report reports/experiment_report_${SCENARIO}_${PROFILE}.json \
#        --db logs/traffic_logs_vlm_${PROFILE}.db \
#        --egress-only

    if [ $? -ne 0 ]; then
        echo ""
        echo "=== Test interrupted at profile: $PROFILE ==="
        echo "=== Remaining profiles skipped ==="
        cleanup
    fi
    echo ""
    echo "[Wait] Waiting ${PROFILE_INTERVAL}s before next profile..."
    sleep ${PROFILE_INTERVAL}
done
echo ""
echo "=== All Done ==="

# ==========================================
# [Added] Archive Results Logic
# ==========================================
ARCHIVE_BASE="archived_results"
RUN_ID=$(date +%Y%m%d_%H%M%S)
TARGET_PATH="${ARCHIVE_BASE}/${RUN_ID}"
ARCHIVE_FILE="${ARCHIVE_BASE}/results_${RUN_ID}.tar.gz"
echo "=== Archiving results to ${TARGET_PATH} ==="
mkdir -p "$TARGET_PATH"
# 1. Archive Capture Dir (--capture-dir)
# Source: results/captures/ (contains all profile subdirs generated in the loop)
if [ -d "results/captures" ]; then
    cp -r results/captures "$TARGET_PATH/captures"
    echo "[Archive] Captures collected"
else
    echo "[Warning] Capture directory not found"
fi
# 2. Archive Reports (--report)
# Source: results/reports (contains all json files generated in the loop)
if [ -d "results/reports" ]; then
    cp -r results/reports "$TARGET_PATH/reports"
    echo "[Archive] Reports collected"
else
    echo "[Warning] Reports directory not found"
fi
# 3. Archive DB (--db)
# Source: logs (contains all .db files generated in the loop)
if [ -d "logs" ]; then
    cp -r logs "$TARGET_PATH/logs"
    echo "[Archive] Logs/DB collected"
else
    echo "[Warning] Logs directory not found"
fi
# -C changes to parent directory so the archive contains relative paths
tar -czf "$ARCHIVE_FILE" -C "$ARCHIVE_BASE" "${RUN_ID}"
if [ $? -eq 0 ]; then
    echo "[Success] Archive created: ${ARCHIVE_FILE}"
    # Optional: Remove uncompressed folder to save space
    # rm -rf "$TARGET_PATH"
else
    echo "[Error] Failed to create archive"
fi
echo "=== Archiving completed ==="
# ==========================================

# clean
sudo pkill -9 -f tcpdump 2>/dev/null
sudo tc qdisc del dev lo root netem 2>/dev/null
sudo tc qdisc del dev lo ingress 2>/dev/null
exit 0
