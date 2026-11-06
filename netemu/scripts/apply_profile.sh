#!/bin/bash
# Apply network emulation profile using tc/netem
# Usage: ./apply_profile.sh <interface> <delay_ms> <loss_pct> [rate_mbit]

set -e

IFACE=${1:-eth0}
DELAY_MS=${2:-0}
LOSS_PCT=${3:-0}
RATE=${4:-}

echo "Applying network profile to $IFACE"
echo "  Delay: ${DELAY_MS}ms"
echo "  Loss: ${LOSS_PCT}%"
echo "  Rate: ${RATE:-unlimited}"

# Clear existing rules
sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true

if [ -z "$RATE" ] || [ "$RATE" = "0" ]; then
    # No rate limiting - use netem only
    if [ "$DELAY_MS" -gt 0 ] || [ "$(echo "$LOSS_PCT > 0" | bc -l)" -eq 1 ]; then
        NETEM_PARAMS=""

        if [ "$DELAY_MS" -gt 0 ]; then
            NETEM_PARAMS="delay ${DELAY_MS}ms"
        fi

        if [ "$(echo "$LOSS_PCT > 0" | bc -l)" -eq 1 ]; then
            NETEM_PARAMS="$NETEM_PARAMS loss ${LOSS_PCT}%"
        fi

        sudo tc qdisc add dev "$IFACE" root netem $NETEM_PARAMS
        echo "Applied netem: $NETEM_PARAMS"
    else
        echo "No impairments to apply"
    fi
else
    # Rate limiting with HTB + netem
    sudo tc qdisc add dev "$IFACE" root handle 1: htb default 11
    sudo tc class add dev "$IFACE" parent 1: classid 1:1 htb rate "${RATE}mbit" ceil "${RATE}mbit"
    sudo tc class add dev "$IFACE" parent 1:1 classid 1:11 htb rate "${RATE}mbit" ceil "${RATE}mbit"

    NETEM_PARAMS=""

    if [ "$DELAY_MS" -gt 0 ]; then
        NETEM_PARAMS="delay ${DELAY_MS}ms"
    fi

    if [ "$(echo "$LOSS_PCT > 0" | bc -l)" -eq 1 ]; then
        NETEM_PARAMS="$NETEM_PARAMS loss ${LOSS_PCT}%"
    fi

    if [ -n "$NETEM_PARAMS" ]; then
        sudo tc qdisc add dev "$IFACE" parent 1:11 handle 10: netem $NETEM_PARAMS
    fi

    echo "Applied HTB rate limiting: ${RATE}mbit"
    echo "Applied netem: $NETEM_PARAMS"
fi

echo "Done. Current tc configuration:"
tc qdisc show dev "$IFACE"
