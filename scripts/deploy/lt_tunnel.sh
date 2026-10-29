#!/usr/bin/env bash
# 本地 web_sim(:8020) -> localtunnel 的公网 HTTPS 隧道守护脚本（固定子域名，断线自动重连）。
#
# 为什么不用 Coolify 反向隧道：容器到 coolify.uwis.cn(43.248.10.52) 的回程 TCP 大包
# 被链路上静默丢弃（服务器发的 43B SSH banner 能到，1120B KEXINIT 到不了），握手卡死。
# 为什么不用 localhost.run：免费版域名会定期变化，无法给服务器 nginx 一个稳定上游。
# localtunnel 免注册即可指定固定子域；首访提醒页由 nginx 加 bypass-tunnel-reminder 头跳过。
#
# 注意：localttunnel 服务端在子域处于「刚断开冷却期」或被占用时会直接分一个随机名，
# 所以这里必须校验拿到的域名，不对就杀掉重试，避免 nginx 上游指向一个错误的地址。
set -u

LOCK_FILE="${LOCK_FILE:-/tmp/lt_tunnel.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] lt tunnel already running"
  exit 0
fi

LOCAL_PORT="${LOCAL_PORT:-8020}"
SUBDOMAIN="${LT_SUBDOMAIN:-uwis-vla-sim}"
WANT="https://${SUBDOMAIN}.loca.lt"
LT_BIN="${LT_BIN:-$(command -v lt || echo /usr/local/nvm/versions/node/v16.20.2/bin/lt)}"
URL_FILE="${URL_FILE:-/tmp/lt_url.txt}"
RAW="${RAW:-/tmp/lt_raw.log}"

echo "$WANT" > "$URL_FILE"
CHILD=""
cleanup() { [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null; }
trap cleanup EXIT
trap 'exit 0' INT TERM

while true; do
  : > "$RAW"
  "$LT_BIN" --port "$LOCAL_PORT" --subdomain "$SUBDOMAIN" >"$RAW" 2>&1 &
  CHILD=$!
  url=""
  for _ in $(seq 1 20); do
    sleep 1
    url="$(grep -o 'https://[a-z0-9-]*\.loca\.lt' "$RAW" | head -1)"
    [ -n "$url" ] && break
    kill -0 "$CHILD" 2>/dev/null || break
  done

  if [ "$url" = "$WANT" ]; then
    echo "[$(date '+%F %T')] tunnel up: $url"
    wait "$CHILD"
    echo "[$(date '+%F %T')] tunnel dropped, retry in 5s"
    sleep 5
  else
    echo "[$(date '+%F %T')] localtunnel gave '${url:-nothing}' (want $WANT), retry in 30s"
    kill "$CHILD" 2>/dev/null
    wait "$CHILD" 2>/dev/null
    sleep 30
  fi
done
