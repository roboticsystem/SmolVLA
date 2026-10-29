#!/usr/bin/env python3
"""record_dataset.py —— 遥操采数：录制 (图像, 激光, 里程计, 遥操动作) → HDF5。

落盘 schema 与训练管线 `export/nav_*.h5` **逐字段对齐**：

    actions (N,2) f32   images (N,H,W,3) u8   lidar (N,120) f32
    states  (N,5) f32   # [x, y, θ, v, ω]
    ep_offset (E+1,) i64    ep_instr (E,) str    ep_success (E,) f32（默认 -1，回填）
    meta: 采集参数、相机内参、归一化统计的来源（不重算！）

**为什么 schema 要对齐**：回工作站后可以直接进 `smolvla_finetune_sim.py` 的
数据类（只差 det_feat 一列——需要用标定后的 HSV refs 重算，见 verify 工具提示）。

两种模式：
    --ros     订阅 /camera/image_raw /scan /odom，发布 /cmd_out（推荐，需要 rclpy）
    （无 ROS） 仅相机 + 指令死推航位，激光全 12.0——只用于走通流程，不可训练

操作（cv2 窗口按键）：
    w/s 前进/后退   a/d 左转/右转   空格 急停
    n  结束当前回合（ep 边界）      q  保存并退出
"""
from __future__ import annotations

import argparse
import fcntl
import os
import select
import sys
import termios
import time
import tty

import numpy as np

try:
    import serial
except ImportError:
    serial = None

try:
    import cv2
except ImportError:
    raise SystemExit("需要 opencv：pip3 install opencv-python")

try:
    import h5py
except ImportError:
    raise SystemExit("需要 h5py：pip3 install h5py")

V_MAX, W_MAX = 0.35, 1.60        # 与训练动作分布一致，勿放大


def resample_lidar(r, n=120, max_range=12.0):
    """任意束数的 LaserScan → 120 维「车头为 0 逆时针」口径（与 vla_bridge 相同）。"""
    r = np.asarray(r, np.float32)
    if r.size == n:
        return np.where(np.isfinite(r), r, max_range)
    idx = np.linspace(0, r.size - 1, n)
    out = np.interp(idx, np.arange(r.size), np.where(np.isfinite(r), r, max_range))
    # 有限束插值会把「超出量程」拉低，这里恢复 out-of-range 常量
    out = np.where(out > max_range, max_range, out)
    return out.astype(np.float32)


# ---- CK 串口遥操（与 CK_ros.py 逐字节一致；合并进本进程，少开一个窗口/进程）----
LOCK_FILE = "/tmp/vla_control.lock"
HEARTBEAT_FILE = "/tmp/ck_ros.heartbeat"
UART_DEV, BAUDRATE = "/dev/ttyS3", 115200
SEND_INTERVAL_S, KEY_RELEASE_S = 0.10, 0.50
PWM_PER_V, PWM_CAP = 123.2, 25
CK_KEYMAP = {  # key → (v_cmd, w_cmd, pwmL, pwmR, dir)，与 CK_simple 同款
    "w": (0.35, 0.0, 25, 25, 1),
    "s": (-0.105, 0.0, 15, 15, 2),
    "a": (0.0, 1.6, 12, 12, 3),
    "d": (0.0, -1.6, 12, 12, 4),
    "x": (0.0, 0.0, 0, 0, 1),
}


def ck_frame(left, right, direction, seq):
    return (b"#M,M," + bytes([left]) + b"," + bytes([right]) + b","
            + bytes([direction]) + bytes([seq]) + b"\r\n")


def ck_stop_frame(seq):
    return ck_frame(0, 0, 1, seq)


def achieved(v_cmd, w_cmd):
    """指令 → 实际达到的速度（PWM cap 后的线性映射，与 CK_ros.py 相同）。"""
    v_ach = 0.0
    if abs(v_cmd) > 0.01:
        pwm = min(PWM_CAP, round(abs(v_cmd) * PWM_PER_V))
        v_ach = (pwm / PWM_PER_V) * (1 if v_cmd > 0 else -1)
    return v_ach, w_cmd            # 转向未封顶（pwm12 < cap），按指令值


class H5Streamer:
    """边采边落盘：数据集按帧扩容写盘，RAM 常驻≈0（2GB 小板不再被撑爆）。"""

    def __init__(self, path, w_req, h_req, frame, ros_mode=False,
                 instruction="<manual>", hfov=-1.0, cam_h=-1.0):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path, self.W_req, self.H_req, self.ros_mode = path, w_req, h_req, bool(ros_mode)
        self.instruction, self.hfov, self.cam_h = instruction, float(hfov), float(cam_h)
        fh_, fw_ = frame.shape[:2]
        self.f = h5py.File(path, "w")
        self.d_img = self.f.create_dataset(
            "images", shape=(0, fh_, fw_, 3), maxshape=(None, fh_, fw_, 3),
            dtype=np.uint8, chunks=(1, fh_, fw_, 3))
        self.d_act = self.f.create_dataset("actions", shape=(0, 2), maxshape=(None, 2), dtype=np.float32)
        self.d_lds = self.f.create_dataset("lidar", shape=(0, 120), maxshape=(None, 120), dtype=np.float32)
        self.d_sts = self.f.create_dataset("states", shape=(0, 5), maxshape=(None, 5), dtype=np.float32)
        self.d_off = self.f.create_dataset("ep_offset", shape=(1,), maxshape=(None,), dtype=np.int64)
        self.offs, self.instrs, self.n = [0], [], 0

    @property
    def episodes(self):
        return len(self.offs) - 1

    def append(self, img, act, ld, st):
        for d, row in ((self.d_img, img), (self.d_act, act),
                       (self.d_lds, ld), (self.d_sts, st)):
            d.resize(self.n + 1, axis=0)
            d[self.n] = row
        self.n += 1

    def end_episode(self):
        if self.n > self.offs[-1] + 5:
            self.offs.append(self.n)
            self.instrs.append(self.instruction)
            return True
        return False

    def close(self):
        e = len(self.offs) - 1
        self.d_off.resize(len(self.offs), axis=0)
        self.d_off[...] = np.asarray(self.offs, np.int64)
        dt2 = h5py.special_dtype(vlen=str)
        instrs = self.instrs + [self.instruction] * (e - len(self.instrs))  # 兜底补齐
        self.f["ep_instr"] = np.asarray(instrs, dtype=dt2)
        self.f["ep_success"] = np.full(e, -1.0, np.float32)   # 遥操后回填 1/0
        self.f.attrs.update({
            "schema": "nav_v1", "ros_mode": int(self.ros_mode),
            "img_w": self.W_req, "img_h": self.H_req,
            "camera_hfov_deg": self.hfov,                      # ← robot.yaml / 命令行自动带入
            "camera_h_m": self.cam_h,
            "instruction": self.instruction,
            "note": "动作=遥操指令；det_feat 需回工作站重算",
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ros", action="store_true", help="ROS2 模式（默认关）")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="data_recorded/session_$(date +%m%d_%H%M).h5")
    ap.add_argument("--img-size", default="640x480", help="落盘分辨率 WxH（原图直存）")
    ap.add_argument("--max-episodes", type=int, default=50)
    ap.add_argument("--instruction", default="",
                    help="本批次所有回合的语言指令（写入 ep_instr；不填则 <manual>，微调前需回填）")
    ap.add_argument("--config", default="../config/robot.yaml",
                    help="robot.yaml 路径：自动读取已标定的相机内参写入 h5 attrs")
    ap.add_argument("--hfov", type=float, default=None, help="覆盖 config 中的 hfov_deg")
    ap.add_argument("--cam-h", type=float, default=None, help="覆盖 config 中的 cam_h")
    ap.add_argument("--headless", action="store_true",
                    help="无 GUI：终端原始模式键盘遥操（SSH/无显示器可用，CK_simple 同款）")
    ap.add_argument("--auto-stop-s", type=float, default=0.5,
                    help="headless 松键自动停车时间（秒）；0=按一下持续走直到按空格/q")
    ap.add_argument("--listen", action="store_true",
                    help="动作来自外部遥操（CK_ros.py 发布的 /cmd_out），本进程不动键盘不发指令")
    ap.add_argument("--teleop", action="store_true",
                    help="合并遥操：本进程直接驱动 /dev/ttyS3（CK 手感），无需再开 CK_ros.py")
    a = ap.parse_args()
    a.out = a.out.replace("$(date +%m%d_%H%M)", time.strftime("%m%d_%H%M"))
    if a.teleop:
        a.headless = True          # 终端原始模式读键（复用 headless 的 tty 逻辑）

    # ---- 相机内参：优先命令行 > robot.yaml > -1（未标定）----
    hfov, cam_h = -1.0, -1.0
    if os.path.exists(a.config):
        try:
            import yaml
            cfg = yaml.safe_load(open(a.config)) or {}
            cam = cfg.get("camera", {})
            hfov = float(cam.get("hfov_deg", -1.0))
            cam_h = float(cam.get("cam_h", -1.0))
            print(f"[config] 从 {a.config} 读到内参：hfov={hfov} cam_h={cam_h}")
        except ImportError:
            print("[WARN] 未装 pyyaml（pip3 install pyyaml），内参不自动回填")
        except Exception as e:                                # noqa: BLE001
            print(f"[WARN] 读 config 失败（{e}），内参不自动回填")
    else:
        print(f"[WARN] 找不到 {a.config}，内参不自动回填")
    if a.hfov is not None:  hfov = a.hfov
    if a.cam_h is not None: cam_h = a.cam_h
    instruction = a.instruction if a.instruction else "<manual>"

    use_ros = False
    node = None
    if a.ros:
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image, LaserScan
            from nav_msgs.msg import Odometry
            from geometry_msgs.msg import Twist

            class Sink(Node):
                def __init__(self):
                    super().__init__("vla_recorder")
                    self.img = self.scan = self.odom = None
                    self.cmd = None                # 外部遥操动作（CK_ros 发布）
                    self.create_subscription(Image, "/camera/image_raw", self._i, 10)
                    self.create_subscription(LaserScan, "/scan", self._s, 10)
                    self.create_subscription(Odometry, "/odom", self._o, 10)
                    self.create_subscription(Twist, "/cmd_out", self._c, 10)

                def _i(self, m):
                    self.img = np.frombuffer(m.data, np.uint8).reshape(
                        m.height, m.width, -1)[:, :, :3].copy()

                def _s(self, m):
                    ang = np.array([m.angle_min + i * m.angle_increment
                                    for i in range(len(m.ranges))])
                    order = np.argsort(((ang + np.pi) % (2 * np.pi)) - np.pi)
                    self.scan = np.asarray(m.ranges, np.float32)[order]

                def _o(self, m):
                    p, q = m.pose.pose.position, m.pose.pose.orientation
                    yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                           1 - 2 * (q.y**2 + q.z**2)))
                    self.odom = np.array([p.x, p.y, yaw,
                                          m.twist.twist.linear.x,
                                          m.twist.twist.angular.z], np.float32)

                def _c(self, m):
                    self.cmd = [float(m.linear.x), float(m.angular.z)]
            rclpy.init()
            node = Sink()
            use_ros = True
            print("[ROS] 订阅 /camera/image_raw /scan /odom")
        except Exception as e:                                # noqa: BLE001
            print(f"[WARN] rclpy 不可用（{e}）→ 退化为相机-only 模式（激光=12，航位死推）")

    ser = lock_fh = None
    if a.teleop:
        if serial is None:
            sys.exit("需要 pyserial：pip3 install pyserial")
        # 单实例锁：保证 ttyS3 同时只有一个写者（与 control_node/CK_ros 互斥）
        lock_fh = open(LOCK_FILE, "w")
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("错误：/dev/ttyS3 已被占用（control_node / 另一个遥操进程），先停掉它们")
        ser = serial.Serial(UART_DEV, BAUDRATE, timeout=0.01)
        print(f"[CK] 已接管 {UART_DEV} @ {BAUDRATE}（w/s/a/d 遥操 x 停，n 分回合，q 保存退出）")

    W, H = map(int, a.img_size.split("x"))
    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    h5w = None                                   # 首帧后建 H5（边采边落盘）
    v_cmd, w_cmd = 0.0, 0.0
    seq, current_key, last_send = 1, None, 0.0
    pose = np.zeros(3, np.float32)                # 死推航位（x,y,θ）
    dt = 1.0 / 10.0
    print(f"指令: {instruction!r}")
    print("w/s 前后  a/d 转向  空格停  n 结束本回合  q 保存退出"
          + ("（headless：松键 0.5s 自动停车）" if a.headless else ""))
    t_last = time.time()

    # ---- headless：终端原始模式读键（CK_simple 同款，SSH 可用）----
    old_tty = None
    if a.headless:
        old_tty = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        print("\r[headless 遥操中…]", flush=True)

    last_key_t = 0.0
    try:
        while h5w is None or h5w.episodes < a.max_episodes:
            ok, frame = cap.read()
            if not ok:
                print("\r抓帧失败，退出", flush=True); break
            if use_ros and node.odom is not None:
                pose = node.odom[:3]
                st = node.odom.copy()
            else:
                # 航位死推：仅无 ROS 模式（误差大，只用于流程验证）
                pose[2] += w_cmd * dt
                pose[0] += v_cmd * np.cos(pose[2]) * dt
                pose[1] += v_cmd * np.sin(pose[2]) * dt
                st = np.array([*pose, v_cmd, w_cmd], np.float32)
            scan = node.scan if (use_ros and node.scan is not None) else np.full(120, 12.0)

            # ---- 读键 / 外部遥操 ----
            v_new, w_new = v_cmd, w_cmd
            quit_req = False
            if a.teleop:
                # 合并遥操：直接驱动 ttyS3（CK 手感），录到的动作即串口指令
                now = time.time()
                with open(HEARTBEAT_FILE, "w") as fh:
                    fh.write(str(now))
                if current_key and now - last_key_t >= KEY_RELEASE_S:
                    current_key = None
                if select.select([sys.stdin], [], [], 0)[0]:
                    for ch in os.read(sys.stdin.fileno(), 64).decode(errors="ignore").lower():
                        if ch in CK_KEYMAP:
                            current_key = ch
                            last_key_t = now
                        elif ch == "n":
                            if h5w is not None and h5w.end_episode():
                                print(f"\r\n  回合 {h5w.episodes} 已记录"
                                      f"（{h5w.offs[-1]-h5w.offs[-2]} 帧）", flush=True)
                            v_new = w_new = 0.0
                        elif ch in ("q", "\x03"):
                            if h5w is not None:
                                h5w.end_episode()
                            quit_req = True
                            break
                if current_key:
                    v_new, w_new = CK_KEYMAP[current_key][:2]
                    if now - last_send >= SEND_INTERVAL_S:
                        _, _, pl, pr, dr = CK_KEYMAP[current_key]
                        ser.write(ck_frame(pl, pr, dr, seq))
                        seq = (seq % 255) + 1
                        last_send = now
                else:
                    v_new = w_new = 0.0
            elif a.listen:
                # 动作来自外部遥操（CK_ros.py 发布的 /cmd_out），本进程不碰键盘遥操
                if node is not None and node.cmd is not None:
                    v_new, w_new = node.cmd
                if a.headless and select.select([sys.stdin], [], [], 0)[0]:
                    for ch in os.read(sys.stdin.fileno(), 64).decode(errors="ignore").lower():
                        if ch == "n":
                            if h5w is not None and h5w.end_episode():
                                print(f"\r\n  回合 {h5w.episodes} 已记录"
                                      f"（{h5w.offs[-1]-h5w.offs[-2]} 帧）", flush=True)
                        elif ch in ("q", "\x03"):
                            if h5w is not None:
                                h5w.end_episode()
                            quit_req = True
                            break
            elif a.headless:
                if select.select([sys.stdin], [], [], 0)[0]:
                    for ch in os.read(sys.stdin.fileno(), 64).decode(errors="ignore").lower():
                        if ch == "w":  v_new = V_MAX
                        elif ch == "s": v_new = -V_MAX * 0.3
                        elif ch == "a": w_new = W_MAX
                        elif ch == "d": w_new = -W_MAX
                        elif ch == " ": v_new = w_new = 0.0
                        elif ch == "n":
                            if h5w is not None and h5w.end_episode():
                                print(f"\r\n  回合 {h5w.episodes} 已记录"
                                      f"（{h5w.offs[-1]-h5w.offs[-2]} 帧）", flush=True)
                            v_new = w_new = 0.0
                        elif ch == "q":
                            if h5w is not None:
                                h5w.end_episode()
                            quit_req = True
                            break
                    if not quit_req:
                        last_key_t = time.time()
                # 松键 auto_stop_s 秒自动停车（防失控；0=不自动停）
                ast = a.auto_stop_s
                if ast > 0 and (v_cmd or w_cmd) and time.time() - last_key_t > ast:
                    v_new = w_new = 0.0
            else:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("w"):  v_new = V_MAX
                elif key == ord("s"): v_new = -V_MAX * 0.3
                elif key == ord("a"): w_new = W_MAX
                elif key == ord("d"): w_new = -W_MAX
                elif key == ord(" "): v_new = w_new = 0.0
                elif key == ord("n"):
                    if h5w is not None and h5w.end_episode():
                        print(f"  回合 {h5w.episodes} 已记录（{h5w.offs[-1]-h5w.offs[-2]} 帧）")
                    v_new = w_new = 0.0
                elif key == ord("q"):
                    if h5w is not None:
                        h5w.end_episode()
                    quit_req = True
            v_cmd, w_cmd = v_new, w_new
            if quit_req:
                break

            if h5w is None:
                h5w = H5Streamer(a.out, W, H, frame, ros_mode=use_ros,
                                 instruction=instruction, hfov=hfov, cam_h=cam_h)
                print(f"\r[H5] 边采边落盘 → {a.out}", flush=True)
            h5w.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), [v_cmd, w_cmd],
                       resample_lidar(scan), st)
            t_last = time.time()

            if a.headless:
                if h5w.n % 10 == 0:
                    print(f"\rep={h5w.episodes} 帧={h5w.n} v={v_cmd:+.2f} w={w_cmd:+.2f}   ",
                          end="", flush=True)
            else:
                view = frame.copy()
                cv2.putText(view, f"ep={h5w.episodes} frames={h5w.n} v={v_cmd:+.2f} w={w_cmd:+.2f}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("record", view)

            if use_ros:
                import rclpy
                rclpy.spin_once(node, timeout_sec=0)
                # teleop 模式：发布 /cmd_vel（实际速度）给 odom_node 积分
                if a.teleop:
                    if not hasattr(node, "pub_vel"):
                        node.pub_vel = node.create_publisher(Twist, "/cmd_vel", 10)
                    va, wa = achieved(v_cmd, w_cmd)
                    m = Twist(); m.linear.x = float(va); m.angular.z = float(wa)
                    node.pub_vel.publish(m)
                # 遥操指令发到底盘（键盘模式才发；listen/teleop 模式直接驱动串口）
                if not a.listen and not a.teleop:
                    try:
                        from geometry_msgs.msg import Twist
                        if not hasattr(node, "pub"):
                            node.pub = node.create_publisher(Twist, "/cmd_out", 10)
                        m = Twist(); m.linear.x = float(v_cmd); m.angular.z = float(w_cmd)
                        node.pub.publish(m)
                    except Exception:
                        pass
    finally:
        if old_tty is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
            print()
        # teleop 收尾：先发停车帧并删心跳（看门狗随后不再介入）
        if a.teleop:
            try:
                ser.write(ck_stop_frame(seq)); ser.close()
            except Exception:                                  # noqa: BLE001
                pass
            try:
                os.remove(HEARTBEAT_FILE)
            except OSError:
                pass

    cap.release()
    if not a.headless:
        cv2.destroyAllWindows()
    n = h5w.n if h5w is not None else 0
    if n < 10:
        if h5w is not None:
            h5w.f.close()
            try:
                os.remove(a.out)
            except OSError:
                pass
        print("样本太少，未保存"); return
    h5w.close()
    sz = os.path.getsize(a.out) / 2**20
    print(f"\n已保存 {a.out}: {h5w.episodes} 回合 / {n} 帧 / {sz:.1f} MB")
    print(f"指令: {instruction!r} | 内参: hfov={hfov} cam_h={cam_h}")
    print(f"下一步：python3 tools/verify_dataset.py --h5 {a.out}")


if __name__ == "__main__":
    main()
