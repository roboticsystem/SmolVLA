#!/usr/bin/env bash
# 本地 web_sim(:8020) -> localhost.run 的公网 HTTPS 隧道守护脚本（断线自动重连）。
#
# 为什么换掉 Coolify 反向隧道：容器到 coolify.uwis.cn(43.248.10.52) 的回程 TCP 大包
# 被链路上静默丢弃（服务器发的 43B SSH banner 能到，1120B KEXINIT 到不了），
# 握手永远卡在 KEXINIT，隧道建不起来；localhost.run 走另一条出口，握手与大数据均正常。
# 当前公网地址写入 URL_FILE，日志里也会打印。
set -u

LOCK_FILE="${LOCK_FILE:-/tmp/lhr_tunnel.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] lhr tunnel already running"
  exit 0
fi

LOCAL_PORT="${LOCAL_PORT:-8020}"
KNOWN_HOSTS="${KNOWN_HOSTS:-/tmp/lhr_known_hosts}"
URL_FILE="${URL_FILE:-/tmp/lhr_url.txt}"
HOST="${LHR_HOST:-localhost.run}"
# 无密钥模式已验证可用（公钥未在 admin.localhost.run 注册时，带 -i 反而不下发域名）
KEY="${LHR_KEY:-}"
KEY_ARGS=()
[ -n "$KEY" ] && KEY_ARGS=(-i "$KEY")

while true; do
  ssh "${KEY_ARGS[@]}" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile="$KNOWN_HOSTS" \
    -o ServerAliveInterval=20 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o LogLevel=ERROR \
    -R "80:localhost:${LOCAL_PORT}" \
    "nokey@${HOST}" 2>&1 | while IFS= read -r line; do
      echo "[$(date '+%F %T')] ${line%$'\r'}"
      case "$line" in
        *lhr.life*tunneled*)
          printf '%s\n' "${line##*https://}" | awk '{print $1}' > "$URL_FILE"
          ;;
      esac
    done
  echo "[$(date '+%F %T')] tunnel exited, retry in 5s"
  sleep 5
done
