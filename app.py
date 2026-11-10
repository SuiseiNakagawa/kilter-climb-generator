"""Setter Lab: a local web app to set climbs, grade them, and generate new ones.

    python app.py              # then open http://localhost:8765 (opens by itself)
    python app.py --port 9000

Runs on your own computer only. Needs the trained models (models/gnn.pt,
models/generator.pt) and data/processed/ from prepare_data.py.

API (JSON):
  GET  /api/board      board positions, angles, grades, reach reference
  POST /api/grade      {angle, holds: [[placement, role], ...]}  -> grade + checks
  POST /api/generate   {angle, grade, temperature, max_move_cm, n} -> candidates
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch

from evaluate_generator import load_generator
from kilter.climb_checks import CHECK_LABELS, CHECKS, Novelty, Plausibility, measurements
from kilter.critic import GradeCritic
from kilter.data import DEFAULT_DB, PROCESSED_DIR, PROJECT_DIR, ROLES, Climb, Hold, load_board, read_processed
from kilter.grades import MAX_DIFFICULTY, MIN_DIFFICULTY, _GRADES
from kilter.sampling import ClimbSampler
from kilter.tokenizer import ClimbTokenizer

CM_PER_UNIT = 2.54                     # Kilter board coordinates appear to be inches
X_MIN, X_MAX, Y_MIN, Y_MAX = 0, 144, 4, 152   # the 12x12 board with kickboard that all training climbs use
PAGE = PROJECT_DIR / "app" / "index.html"
LABELS = {**CHECK_LABELS, "move_max": "biggest hand gap"}


class Lab:
    def __init__(self):
        t = time.time()
        print("loading board, models and training climbs (about 20 s)...")
        self.board = load_board()
        with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as con:
            sets = dict(con.execute("SELECT id, set_id FROM placements WHERE layout_id = 1"))
        self.positions = [{"id": p, "x": x, "y": y, "small": sets.get(p) == 20}
                          for p, (x, y) in sorted(self.board.items(), key=lambda kv: (kv[1][1], kv[1][0]))
                          if X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX]
        self.tok = ClimbTokenizer.load(PROCESSED_DIR / "tokenizer.json")
        train = read_processed("train")
        self.plaus, self.novelty = Plausibility(train), Novelty(train)
        self.critic = GradeCritic()
        self.sampler = ClimbSampler(load_generator(), self.tok, self.board)
        by_grade = defaultdict(list)
        for c in train:
            by_grade[int(c.grade[1:])].append(measurements(c)["move_max"] * CM_PER_UNIT)
        # distribution of each grade's biggest hand gap, as percentiles 0, 5, ..., 100
        self.reach_ref = {g: {"median": round(float(np.median(v))), "p90": round(float(np.percentile(v, 90))),
                              "pct": [round(float(x), 1) for x in np.percentile(v, np.arange(0, 101, 5))]}
                          for g, v in sorted(by_grade.items()) if g <= 10}
        self.lock = threading.Lock()
        print(f"ready in {time.time() - t:.0f}s")

    # ---- helpers
    @staticmethod
    def grade_names(difficulty: float) -> tuple[int, str]:
        d = int(np.clip(round(difficulty), MIN_DIFFICULTY, MAX_DIFFICULTY))
        font, v = _GRADES[d]
        return int(v[1:]), font

    def analyse(self, climbs: list[Climb]) -> list[dict]:
        diff = self.critic.predict(climbs)
        checks = self.plaus.check(climbs)
        overlap, match = self.novelty.nearest(climbs)
        out = []
        for c, d, ok, ov, m in zip(climbs, diff, checks, overlap, match):
            meas = measurements(c)
            v, font = self.grade_names(float(d))
            out.append({
                "holds": [[h.placement, h.role] for h in c.holds],
                "angle": c.angle,
                "difficulty": round(float(d), 2), "v": v, "font": font,
                "move_max_cm": round(meas["move_max"] * CM_PER_UNIT),
                "checks": [{"key": k, "label": LABELS[k], "ok": bool(o),
                            "value": self._fmt(k, meas[k]),
                            "range": f"{self._fmt(k, self.plaus.lo[i])}–{self._fmt(k, self.plaus.hi[i])}"}
                           for i, (k, o) in enumerate(zip(CHECKS, ok))],
                "plausible": bool(ok.all()),
                "closest": {"name": m.name, "overlap": round(float(ov), 3), "grade": m.grade, "angle": m.angle},
            })
        return out

    @staticmethod
    def _fmt(key: str, v: float) -> str:
        if np.isnan(v):
            return "–"
        if key in ("n_holds", "n_hand"):
            return str(int(round(v)))
        return f"{round(v * CM_PER_UNIT)} cm"

    def to_climb(self, holds, angle) -> Climb:
        hs = sorted((Hold(int(p), r, *self.board[int(p)]) for p, r in holds if int(p) in self.board and r in ROLES),
                    key=lambda h: (h.y, h.x))
        return Climb(uuid="", name="", angle=int(angle), difficulty=float("nan"), grade="",
                     font_grade="", ascents=0, quality=0.0, holds=hs)

    # ---- endpoints
    def board_info(self) -> dict:
        return {"positions": self.positions, "angles": self.tok.angles,
                "grades": list(range(0, 11)), "reach_ref": self.reach_ref,
                "cm_per_unit": CM_PER_UNIT, "bounds": [X_MIN, X_MAX, Y_MIN, Y_MAX]}

    def grade(self, req: dict) -> dict:
        c = self.to_climb(req.get("holds", []), req.get("angle", 40))
        roles = [h.role for h in c.holds]
        warnings = []
        if roles.count("start") == 0:
            warnings.append("Add a start hold.")
        if roles.count("finish") == 0:
            warnings.append("Add a finish hold.")
        if roles.count("start") > 2 or roles.count("finish") > 2:
            warnings.append("Real climbs have at most 2 start and 2 finish holds.")
        if len(c.holds) < 2:
            return {"empty": True, "warnings": warnings}
        with self.lock:
            res = self.analyse([c])[0]
        res["warnings"] = warnings
        return res

    def generate(self, req: dict) -> dict:
        angle, grade = int(req.get("angle", 40)), int(req.get("grade", 3))
        temp = float(req.get("temperature", 0.8))
        n = int(min(max(req.get("n", 48), 8), 128))
        max_cm = req.get("max_move_cm")
        max_move = float(max_cm) / CM_PER_UNIT if max_cm else None
        t = time.time()
        with self.lock:
            samples = self.sampler.sample([(angle, f"V{grade}")] * n, temperature=temp,
                                          seed=int(time.time() * 1000) % 2**31, max_move=max_move)
            done = [s for s in samples if s.wellformed]
            seen, uniq = set(), []
            for s in done:
                key = frozenset((h.placement, h.role) for h in s.holds)
                if key not in seen:
                    seen.add(key)
                    uniq.append(s)
            res = self.analyse([s.as_climb() for s in uniq]) if uniq else []
        for r in res:
            r["off_by"] = r["v"] - grade
        # plausible first, then closest to the requested grade, then most novel
        res.sort(key=lambda r: (not r["plausible"], abs(r["off_by"]), r["closest"]["overlap"]))
        return {"requested": {"angle": angle, "grade": grade, "temperature": temp, "max_move_cm": max_cm},
                "n_sampled": n, "n_complete": len(done), "n_unique": len(uniq),
                "n_plausible": sum(r["plausible"] for r in res),
                "n_on_grade": sum(r["plausible"] and abs(r["off_by"]) <= 1 for r in res),
                "candidates": res[:12], "seconds": round(time.time() - t, 1)}


def make_handler(lab: Lab):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/board":
                self._json(lab.board_info())
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/grade":
                    self._json(lab.grade(req))
                elif self.path == "/api/generate":
                    self._json(lab.generate(req))
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:  # show the problem in the page instead of hanging
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def log_message(self, fmt, *args):
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    lab = Lab()
    server = HTTPServer(("127.0.0.1", args.port), make_handler(lab))
    url = f"http://localhost:{args.port}"
    print(f"Setter Lab running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
