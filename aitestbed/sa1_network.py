"""Private, bidirectionally shaped Internet path for a single provider job.

Only task-owned veth devices are shaped. Host forwarding policy and physical
interface qdiscs are preserved; narrowly scoped forwarding/NAT rules are removed
on exit. A fresh client namespace also resets its TCP path metrics each time.
"""

import ipaddress
import json
from pathlib import Path
import shlex
import subprocess
import uuid

from netemu.tc import shaping_commands


def command(args, *, check=True):
    return subprocess.run(args, check=check, text=True, capture_output=True)


class IsolatedPath:
    def __init__(self, profile, interface, subnet="10.249.240.0/29"):
        network = ipaddress.ip_network(subnet)
        if network.version != 4 or network.prefixlen != 29:
            raise ValueError("An IPv4 /29 is required")
        self.network = network
        self.client_ip = str(network[2])
        self.lan_ip = str(network[1])
        self.host_ip = str(network[5])
        self.wan_ip = str(network[6])
        self.lan_subnet = str(next(network.subnets(new_prefix=30)))
        token = uuid.uuid4().hex[:8]
        self.client = "sa1c-" + token
        self.router = "sa1r-" + token
        self.host_device = "s1h" + token
        self.profile, self.interface = profile, interface
        self.undo = []
        self.evidence = {
            "client_namespace": self.client,
            "router_namespace": self.router,
            "host_device": self.host_device,
            "physical_interface": interface,
            "client_ip": self.client_ip,
            "capture_points": {
                "client": "eth0 before UL / after DL impairment",
                "edge": "host veth after UL / before DL impairment",
            },
            "public_provider_one_way_latency_available": False,
        }

    def root(self, args):
        return command(["sudo", "-n", *args])

    def ns(self, namespace, args):
        return self.root(["ip", "netns", "exec", namespace, *args])

    def add(self, args, undo):
        self.root(args)
        self.undo.append(undo)

    def setup(self):
        routes = json.loads(
            command(["ip", "-j", "route", "show", "table", "all"]).stdout
        )
        for route in routes:
            dst = route.get("dst", "default")
            if dst != "default" and ipaddress.ip_network(dst, strict=False).overlaps(
                self.network
            ):
                raise RuntimeError("Private campaign subnet overlaps an existing route")
        if command(["sysctl", "-n", "net.ipv4.ip_forward"]).stdout.strip() != "1":
            raise RuntimeError("Host IPv4 forwarding must already be enabled")
        self.evidence["physical_qdisc_before"] = json.loads(
            self.root(["tc", "-j", "qdisc", "show", "dev", self.interface]).stdout
        )
        try:
            for ns in (self.client, self.router):
                self.add(["ip", "netns", "add", ns], ["ip", "netns", "del", ns])
                self.ns(ns, ["ip", "link", "set", "lo", "up"])
            self.add(
                [
                    "ip",
                    "link",
                    "add",
                    self.host_device,
                    "type",
                    "veth",
                    "peer",
                    "name",
                    "wan",
                    "netns",
                    self.router,
                ],
                ["ip", "link", "del", self.host_device],
            )
            self.ns(
                self.router,
                [
                    "ip",
                    "link",
                    "add",
                    "lan",
                    "type",
                    "veth",
                    "peer",
                    "name",
                    "eth0",
                    "netns",
                    self.client,
                ],
            )
            for ns, dev, address in (
                (None, self.host_device, self.host_ip),
                (self.router, "wan", self.wan_ip),
                (self.router, "lan", self.lan_ip),
                (self.client, "eth0", self.client_ip),
            ):
                prefix = ["ip", "netns", "exec", ns] if ns else []
                self.root(
                    [*prefix, "ip", "address", "add", address + "/30", "dev", dev]
                )
                self.root([*prefix, "ip", "link", "set", dev, "mtu", "1500", "up"])
                self.root(
                    [
                        *prefix,
                        "ethtool",
                        "-K",
                        dev,
                        "tso",
                        "off",
                        "gso",
                        "off",
                        "gro",
                        "off",
                        "lro",
                        "off",
                    ]
                )
                features = self.root([*prefix, "ethtool", "-k", dev]).stdout
                for feature in (
                    "tcp-segmentation-offload",
                    "generic-segmentation-offload",
                    "generic-receive-offload",
                    "large-receive-offload",
                ):
                    if feature + ": off" not in features:
                        raise RuntimeError("Packet coalescing was not disabled")
                self.evidence.setdefault("offloads", {})[f"{ns or 'host'}/{dev}"] = (
                    features
                )
            self.ns(self.router, ["sysctl", "-qw", "net.ipv4.ip_forward=1"])
            self.ns(self.router, ["ip", "route", "add", "default", "via", self.host_ip])
            self.ns(self.client, ["ip", "route", "add", "default", "via", self.lan_ip])
            route = [self.lan_subnet, "via", self.wan_ip, "dev", self.host_device]
            self.add(["ip", "route", "add", *route], ["ip", "route", "del", *route])
            # The host may have Docker's default DROP forwarding policy.
            rules = [
                (
                    [],
                    "FORWARD",
                    [
                        "-i",
                        self.host_device,
                        "-o",
                        self.interface,
                        "-s",
                        self.client_ip + "/32",
                        "-j",
                        "ACCEPT",
                    ],
                ),
                (
                    [],
                    "FORWARD",
                    [
                        "-i",
                        self.interface,
                        "-o",
                        self.host_device,
                        "-d",
                        self.client_ip + "/32",
                        "-m",
                        "conntrack",
                        "--ctstate",
                        "ESTABLISHED,RELATED",
                        "-j",
                        "ACCEPT",
                    ],
                ),
                (
                    ["-t", "nat"],
                    "POSTROUTING",
                    [
                        "-s",
                        self.client_ip + "/32",
                        "-o",
                        self.interface,
                        "-j",
                        "MASQUERADE",
                    ],
                ),
            ]
            for table, chain, rule in rules:
                rule = [
                    *rule[:-2],
                    "-m",
                    "comment",
                    "--comment",
                    self.client,
                    *rule[-2:],
                ]
                self.add(
                    ["iptables", "-w", *table, "-I", chain, "1", *rule],
                    ["iptables", "-w", *table, "-D", chain, *rule],
                )
            dns_dir = "/etc/netns/" + self.client
            if not Path("/etc/netns").exists():
                self.add(["mkdir", "/etc/netns"], ["rmdir", "/etc/netns"])
            self.add(["mkdir", dns_dir], ["rmdir", dns_dir])
            # Avoid the host's loopback-only resolved stub within the namespace.
            source = "/run/systemd/resolve/resolv.conf"
            if not Path(source).exists():
                source = "/etc/resolv.conf"
            self.add(
                ["cp", source, dns_dir + "/resolv.conf"],
                ["rm", dns_dir + "/resolv.conf"],
            )
            self.evidence["commands"] = {}
            for dev, profile in (
                ("wan", self.profile.uplink or self.profile),
                ("lan", self.profile),
            ):
                commands = shaping_commands(profile, dev)
                self.evidence["commands"][dev] = commands
                for cmd in commands:
                    self.ns(self.router, shlex.split(cmd)[1:])
            self.evidence["qdisc_before"] = self.qdiscs()
            return self
        except BaseException:
            self.close()
            raise

    def qdiscs(self):
        return {
            dev: {
                "qdisc": json.loads(
                    self.ns(
                        self.router, ["tc", "-s", "-j", "qdisc", "show", "dev", dev]
                    ).stdout
                ),
                "class_text": self.ns(
                    self.router, ["tc", "-s", "class", "show", "dev", dev]
                ).stdout,
            }
            for dev in ("wan", "lan")
        }

    def close(self):
        failures = []
        while self.undo:
            args = self.undo.pop()
            completed = command(["sudo", "-n", *args], check=False)
            if completed.returncode:
                failures.append({"command": args, "stderr": completed.stderr})
        self.evidence["cleanup_errors"] = failures
        after = json.loads(
            self.root(["tc", "-j", "qdisc", "show", "dev", self.interface]).stdout
        )
        self.evidence["physical_qdisc_after"] = after
        self.evidence["physical_qdisc_unchanged"] = after == self.evidence.get(
            "physical_qdisc_before"
        )
        if failures:
            raise RuntimeError(
                "Private network cleanup failed; inspect network evidence"
            )

    def __enter__(self):
        return self.setup()

    def __exit__(self, *exc):
        self.close()
