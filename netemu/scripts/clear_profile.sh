#!/bin/bash
# Clear all tc/netem rules from an interface
# Usage: ./clear_profile.sh [interface]

IFACE=${1:-eth0}

echo "Clearing tc rules from $IFACE"

sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true

echo "Done. Current tc configuration:"
tc qdisc show dev "$IFACE"
