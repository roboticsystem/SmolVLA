#!/usr/bin/env python3
"""仿真态势网页服务：订阅 /scan 与 /odom，生成 SVG 俯视图。

用法：source docker/ros2_env.sh && python3 docker/viz_server.py
浏览器打开 http://localhost:8000/
"""
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

lock = threading.Lock()
state = {"scan": None, "odom": None, "trail": []}

SCAN_TOPIC = "/scan"
ODOM_TOPIC = "/odom"
WORLD = 14.0          # 显示范围 ±7m
SIZE = 640            # 画布像素

HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>TurtleBot3 仿真态势</title>
<style>body{background:#111;color:#ccc;font-family:monospace;text-align:center}
h3{margin:4px}#info{font-size:13px;color:#8f8}</style></head><body>
<h3>TurtleBot3 Gazebo 仿真 · 俯视态势（/scan + /odom）</h3>
<div id="img"></div>
<div id="info">connecting...</div>
<script>
async function refresh(){
  try{
    const r = await fetch('/scene.svg?t='+Date.now());
    document.getElementById('img').innerHTML = await r.text();
  }catch(e){ document.getElementById('info').textContent='server offline'; }
}
setInterval(refresh, 400); refresh();
</script></body></html>"""


class Sub(Node):
    def __init__(self):
        super().__init__("viz_server_sub")
        self.create_subscription(LaserScan, SCAN_TOPIC, self.on_scan, 5)
        self.create_subscription(Odometry, ODOM_TOPIC, self.on_odom, 5)

    def on_scan(self, m):
        with lock:
            state["scan"] = (m.angle_min, m.angle_increment, list(m.ranges))

    def on_odom(self, m):
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with lock:
            state["odom"] = (p.x, p.y, yaw)
            trail = state["trail"]
            if not trail or math.hypot(p.x - trail[-1][0], p.y - trail[-1][1]) > 0.05:
                trail.append((p.x, p.y))
                if len(trail) > 600:
                    state["trail"] = trail[::2]


def to_px(x, y):
    s = SIZE / WORLD
    return SIZE / 2 + x * s, SIZE / 2 - y * s


def scene_svg():
    with lock:
        scan = state["scan"]
        odom = state["odom"]
        trail = list(state["trail"])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" '
             f'style="background:#181818;border:1px solid #333">']
    # 网格（1m）
    for i in range(-int(WORLD // 2), int(WORLD // 2) + 1):
        x1, y1 = to_px(i, -WORLD / 2); x2, y2 = to_px(i, WORLD / 2)
        parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="#262626"/>')
        x1, y1 = to_px(-WORLD / 2, i); x2, y2 = to_px(WORLD / 2, i)
        parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="#262626"/>')

    # 激光点
    if scan:
        a0, da, rng = scan
        ox = oy = 0.0; yaw = 0.0
        if odom:
            ox, oy, yaw = odom
        for i, r in enumerate(rng):
            if not (0.12 < r < 12.0):
                continue
            a = a0 + i * da + yaw
            x, y = ox + r * math.cos(a), oy + r * math.sin(a)
            px, py = to_px(x, y)
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.6" fill="#e55"/>')

    # 轨迹
    if len(trail) > 1:
        pts = " ".join(f"{to_px(x, y)[0]:.1f},{to_px(x, y)[1]:.1f}" for x, y in trail)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#58f" stroke-width="1.5"/>')

    # 机器人（三角）
    if odom:
        x, y, yaw = odom
        px, py = to_px(x, y)
        s = SIZE / WORLD
        L = 14
        dx, dy = math.cos(yaw), math.sin(yaw)
        nx, ny = -dy, dx
        p1 = (px + dx * L, py - dy * L)
        p2 = (px - dx * L * 0.7 + nx * L * 0.6, py + dy * L * 0.7 + ny * L * 0.6)
        p3 = (px - dx * L * 0.7 - nx * L * 0.6, py + dy * L * 0.7 - ny * L * 0.6)
        parts.append(f'<polygon points="{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f} '
                     f'{p3[0]:.1f},{p3[1]:.1f}" fill="#5f5"/>')
        parts.append(f'<text x="{px + 16:.0f}" y="{py - 12:.0f}" fill="#8f8" font-size="12">'
                     f'({x:.2f}, {y:.2f})</text>')
    parts.append("</svg>")
    return "".join(parts)


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/scene.svg"):
            self._send(scene_svg().encode(), "image/svg+xml")
        elif self.path.startswith("/status"):
            with lock:
                s = bool(state["scan"]); o = bool(state["odom"])
            self._send(f'{{"scan":{str(s).lower()},"odom":{str(o).lower()}}}'.encode(),
                       "application/json")
        else:
            self._send(HTML.encode(), "text/html")

    def log_message(self, *a):
        pass


def main():
    rclpy.init()
    node = Sub()
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    srv = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)
    print("viz server on :8000, topics /scan + /odom", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
