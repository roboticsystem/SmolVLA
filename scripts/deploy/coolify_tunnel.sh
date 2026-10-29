#!/usr/bin/env bash
# 本地 web_sim(:8020) -> Coolify 主机 127.0.0.1:18021 的反向隧道守护脚本。
# 服务器端 nginx(:18020, Basic Auth) 反代到该隧道；断线自动重连。
set -u

# 同一时间只允许一个守护进程，供 .bashrc 开机自启时幂等调用
LOCK_FILE="${LOCK_FILE:-/tmp/coolify_tunnel.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] tunnel already running"
  exit 0
fi

HOST="${COOLIFY_HOST:-root@coolify.uwis.cn}"
REMOTE_PORT="${REMOTE_PORT:-18021}"
LOCAL_PORT="${LOCAL_PORT:-8020}"
KNOWN_HOSTS="${KNOWN_HOSTS:-/tmp/vla_known_hosts}"

CHILD_PID=""
cleanup() {
  [ -n "$CHILD_PID" ] && kill "$CHILD_PID" 2>/dev/null
}
trap cleanup EXIT
trap 'exit 0' INT TERM

while true; do
  ssh -N \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=20 \
    -o ServerAliveCountMax=3 \
    -o UserKnownHostsFile="$KNOWN_HOSTS" \
    -o StrictHostKeyChecking=accept-new \
    -o LogLevel=ERROR \
    -R "127.0.0.1:${REMOTE_PORT}:127.0.0.1:${LOCAL_PORT}" \
    "$HOST" &
  CHILD_PID=$!
  wait "$CHILD_PID"
  echo "[$(date '+%F %T')] tunnel exited, retry in 5s"
  sleep 5
done
