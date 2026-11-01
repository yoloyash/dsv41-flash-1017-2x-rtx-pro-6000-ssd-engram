#!/usr/bin/env bash
# Run serve-ssd.sh as a systemd service, so the server comes back after a reboot or a crash. Opt-in: it holds both GPUs and up to
# CONTAINER_MEM of RAM from every boot. serve-ssd.sh itself stays a foreground command.
#   sudo serve/install-service.sh API_KEY=change-me PACK=/models/dsv41-1017 [PORT=8000 HOST_CACHE=512 ...]   any serve-ssd.sh knob
#   sudo serve/install-service.sh --remove
# Starts it now. Settings live in /etc/dsv41-flash.env (root, 0600): edit, then systemctl restart dsv41-flash.
# Not --restart unless-stopped: serve.sh runs docker with --rm, and its warmup runs next to the container, not inside it.
#   journalctl -fu dsv41-flash          loading takes minutes; "[warmup] done" is the signal
#   sudo systemctl stop dsv41-flash     frees the GPUs until the next boot; disable keeps them free across boots
set -eu
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNIT=/etc/systemd/system/dsv41-flash.service; ENV_FILE=/etc/dsv41-flash.env
[[ $EUID == 0 ]] || { echo "run with sudo"; exit 1; }
if [[ ${1:-} == --remove ]]; then
  systemctl disable --now dsv41-flash 2>/dev/null || true
  rm -f "$UNIT" "$ENV_FILE"; systemctl daemon-reload; echo "removed"; exit 0
fi
RUN_USER=${SUDO_USER:?run with sudo from your own account: the service runs as that user}
id -nG "$RUN_USER" | grep -qw docker || { echo "$RUN_USER is not in the docker group"; exit 1; }
API_KEY=; PACK=
for kv in "$@"; do
  [[ $kv =~ ^[A-Za-z_][A-Za-z0-9_]*=[^\"\\]*$ ]] || { echo "bad setting (want KEY=value, no quotes or backslashes): ${kv%%=*}"; exit 1; }
  case $kv in API_KEY=*) API_KEY=${kv#*=} ;; PACK=*) PACK=${kv#*=} ;; esac
done
[[ -n $API_KEY ]] || { echo "API_KEY=... is required"; exit 1; }
[[ -f $PACK/config.json ]] || { echo "PACK=... must be the downloaded pack (no config.json in '$PACK')"; exit 1; }
sudo -u "$RUN_USER" test -r "$ROOT/serve/serve-ssd.sh" || { echo "$RUN_USER can't read $ROOT"; exit 1; }
( umask 077; for kv in "$@"; do printf '%s="%s"\n' "${kv%%=*}" "${kv#*=}"; done > "$ENV_FILE" )
cat > "$UNIT" <<UNIT_EOF
[Unit]
Description=DeepSeek-V4.1-Flash-1017 with Engram tables on SSD ($ROOT)
Requires=docker.service
Wants=network-online.target
After=docker.service network-online.target
RequiresMountsFor=$PACK
# A load takes minutes; stop retrying after three failures in an hour
StartLimitIntervalSec=3600
StartLimitBurst=3

[Service]
User=$RUN_USER
WorkingDirectory=$ROOT
EnvironmentFile=$ENV_FILE
ExecStartPre=-/bin/sh -c 'docker rm -f "\$\${NAME:-dsv41-flash-exl3}" >/dev/null 2>&1'
ExecStart=/bin/bash $ROOT/serve/serve-ssd.sh
ExecStopPost=-/bin/sh -c 'docker rm -f "\$\${NAME:-dsv41-flash-exl3}" >/dev/null 2>&1'
Restart=on-failure
RestartSec=30
TimeoutStopSec=120
SuccessExitStatus=130 143

[Install]
WantedBy=multi-user.target
UNIT_EOF
systemctl daemon-reload
systemctl enable dsv41-flash
systemctl restart dsv41-flash
echo "dsv41-flash installed and starting; watch it with: journalctl -fu dsv41-flash"
