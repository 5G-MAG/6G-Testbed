"""Build tc commands from runtime profiles without modifying the host."""

import shlex

from .profile import NetworkProfile


def netem_parameters(profile: NetworkProfile) -> list[str]:
    """Preserve legacy syntax and explicit calibration rate/loss semantics."""
    p = profile
    p.validate()
    if p.bypass:
        return []
    params = []
    limit = p.limit_packets
    if limit is None and p.rate_mode is not None:
        limit = 1000  # Same explicit default as the calibration command builder.
    if limit is not None:
        params.append(f"limit {int(limit)}")
    if p.delay_ms:
        delay = f"delay {p.delay_ms}ms"
        if p.jitter_ms:
            delay += f" {p.jitter_ms}ms"
            if p.delay_correlation_pct is not None:
                delay += f" {p.delay_correlation_pct}%"
            # Uniform is the kernel's built-in distribution, not a table file.
            if p.delay_distribution not in (None, "none", "fixed", "uniform"):
                delay += f" distribution {p.delay_distribution}"
        params.append(delay)
    if p.loss_model == "gemodel" and p.loss_mean_burst_packets is not None:
        enter, leave = p.ge_probabilities()
        params.append(f"loss gemodel {enter:.12g}% {leave:.12g}% 100% 0%")
    elif p.loss_model in {"gemodel", "state"}:
        # Preserve older flat profiles that express tc's transition probability.
        loss = f"loss {p.loss_model}"
        if p.loss_pct:
            loss += f" {p.loss_pct}%"
            if p.loss_correlation_pct is not None:
                loss += f" {p.loss_correlation_pct}%"
        params.append(loss)
    elif p.loss_pct:
        loss = "loss random" if p.loss_model == "random" else "loss"
        loss += f" {p.loss_pct}%"
        if p.loss_correlation_pct is not None:
            loss += f" {p.loss_correlation_pct}%"
        params.append(loss)
    for keyword, value, correlation in (
        ("corrupt", p.corruption_pct, p.corruption_correlation_pct),
        ("duplicate", p.duplicate_pct, p.duplicate_correlation_pct),
        ("reorder", p.reorder_pct,
         p.reorder_correlation_pct if p.reorder_correlation_pct is not None else 25),
    ):
        if value:
            option = f"{keyword} {value}%"
            if correlation is not None:
                option += f" {correlation}%"
            params.append(option)
    rate = None
    if p.rate_mode == "netem":
        rate = p.rate_mbit
    elif p.rate_mode == "htb_netem":
        rate = p.netem_rate_mbit or p.rate_ceil_mbit or p.rate_mbit
    if rate is not None:
        params.append(f"rate {rate}mbit")
    if p.slot_distribution:
        slot = (f"slot distribution {p.slot_distribution} "
                f"{p.slot_delay_ms}ms {p.slot_jitter_ms}ms")
        if p.slot_max_packets is not None:
            slot += f" packets {int(p.slot_max_packets)}"
        if p.slot_max_bytes is not None:
            slot += f" bytes {int(p.slot_max_bytes)}"
        params.append(slot)
    return params


def shaping_commands(profile: NetworkProfile, device: str, *, parent: str | None = None,
                     htb_handle: int = 1, netem_handle: int = 10) -> list[str]:
    """Build an entire service model, at the root or under a loopback band."""
    p = profile
    params = netem_parameters(p)
    if p.bypass:
        return []
    dev = shlex.quote(device)
    attachment = f"parent {parent}" if parent else "root"
    mode = p.rate_mode or ("htb" if p.rate_mbit is not None else "netem")
    if mode == "netem":
        return ([f"sudo tc qdisc add dev {dev} {attachment} handle {netem_handle}: "
                 f"netem {' '.join(params)}"] if params else [])
    rate = f"htb rate {p.rate_mbit}mbit ceil {p.rate_ceil_mbit or p.rate_mbit}mbit"
    for name, byte_count, kbits in (("burst", p.burst_bytes, p.rate_burst_kbit),
                                   ("cburst", p.cburst_bytes, p.rate_cburst_kbit)):
        if byte_count is not None:
            rate += f" {name} {int(byte_count)}"
        elif kbits is not None:
            rate += f" {name} {kbits}kbit"
    commands = [
        f"sudo tc qdisc add dev {dev} {attachment} handle {htb_handle}: htb default 11",
        f"sudo tc class add dev {dev} parent {htb_handle}: classid {htb_handle}:1 {rate}",
        f"sudo tc class add dev {dev} parent {htb_handle}:1 classid {htb_handle}:11 {rate}",
    ]
    if params:
        commands.append(f"sudo tc qdisc add dev {dev} parent {htb_handle}:11 "
                        f"handle {netem_handle}: netem {' '.join(params)}")
    return commands
