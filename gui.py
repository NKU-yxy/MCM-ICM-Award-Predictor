#!/usr/bin/env python3
"""
MCM/ICM 论文获奖预测器  ·  PIXEL-RETRO GUI
═══════════════════════════════════════════
Inspired by 8-bit / pixel-art aesthetics:
  • Monospace "bitmap" fonts (Courier New / Consolas)
  • Blocky UI elements, hard edges, no rounded corners
  • Yellow/gold banner, cyan accent buttons
  • ASCII-art dividers & retro status text
"""

import os, sys, threading, tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
#                        PIXEL DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════
TITLE    = "MCM / ICM  AWARD  PREDICTOR"
WIN_W, WIN_H = 960, 780

# Palette
BG          = "#1a1a2e"        # deep navy background
CARD_BG     = "#16213e"        # dark blue-grey cards
CARD_BORDER = "#0f3460"        # card outline
BANNER_BG   = "#e7b416"        # golden-yellow banner
BANNER_FG   = "#1a1a2e"        # banner text (dark)

ACCENT      = "#00d2d3"        # cyan / aqua buttons
ACCENT_HOVER= "#01a3a4"        # darker cyan hover
ACCENT_LIGHT= "#0a3d62"        # subtle accent bg

TEXT_PRI    = "#eaeaea"         # primary text (near white)
TEXT_SEC    = "#a0a0b0"         # secondary text
TEXT_TER    = "#6a6a80"         # tertiary / muted

GREEN       = "#00e676"         # good / pass
GREEN_DARK  = "#00c853"
ORANGE      = "#ffa726"
RED         = "#ff5252"
YELLOW      = "#e7b416"

PIXEL_CHAR  = "█"              # block char for bars

PROBLEMS = ["自动检测", "A", "B", "C", "D", "E", "F"]


def _pfont(size, weight="normal"):
    """Pixel monospace font."""
    return ("Courier New", size, weight)


def _pfont_bold(size):
    return _pfont(size, "bold")


# ═══════════════════════════════════════════════════════════════════════════
class PredictorGUI:
    """8-bit pixel-style predictor interface."""

    def __init__(self):
        if DND_AVAILABLE:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()

        self.root.title("MCM/ICM Predictor ▪ Pixel Edition")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.configure(bg=BG)
        self.root.minsize(880, 700)

        # State
        self.pdf_path_var = tk.StringVar()
        self.problem_var  = tk.StringVar(value="自动检测")
        self.year_var     = tk.StringVar(value="")
        self.predictor    = None
        self.result_data  = None

        self._style()
        self._build()

    # ── ttk style ──────────────────────────────────────────────────────
    def _style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("P.TCombobox",
                    fieldbackground="#0f3460", background="#0f3460",
                    foreground=TEXT_PRI, arrowcolor=ACCENT,
                    borderwidth=2, relief="solid", padding=4,
                    selectbackground="#0f3460", selectforeground=TEXT_PRI)
        s.map("P.TCombobox",
              fieldbackground=[("readonly", "#0f3460")],
              selectbackground=[("readonly", "#0f3460")],
              selectforeground=[("readonly", TEXT_PRI)])
        s.configure("Vertical.TScrollbar",
                    gripcount=0, background=CARD_BORDER,
                    troughcolor=BG, borderwidth=0)

    # ── main layout ───────────────────────────────────────────────────
    def _build(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview,
                            style="Vertical.TScrollbar")
        self.inner = tk.Frame(self.canvas, bg=BG)

        self.inner.bind("<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win_id = self.canvas.create_window((0, 0), window=self.inner, anchor="n")

        def _recentre(event):
            self.canvas.itemconfigure(self._win_id, width=event.width)
        self.canvas.bind("<Configure>", _recentre)

        self.canvas.configure(yscrollcommand=vsb.set)
        self.canvas.bind_all("<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self.canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Content wrapper
        self.wrap = tk.Frame(self.inner, bg=BG)
        self.wrap.pack(pady=0, padx=40, fill="x")

        self._banner()
        self._ascii_divider("═")
        self._input_card()
        self._action_bar()
        self._ascii_divider("─")
        self._result_frame()

    # ── pixel banner ───────────────────────────────────────────────────
    def _banner(self):
        ban = tk.Frame(self.wrap, bg=BANNER_BG)
        ban.pack(fill="x", pady=(20, 0))

        inner = tk.Frame(ban, bg=BANNER_BG)
        inner.pack(fill="x", padx=24, pady=14)

        # Pixel art star
        star = "★"
        tk.Label(inner, text=star, font=_pfont(18),
                 fg="#1a1a2e", bg=BANNER_BG).pack(side="left", padx=(0, 12))

        col = tk.Frame(inner, bg=BANNER_BG)
        col.pack(side="left", fill="x", expand=True)

        tk.Label(col, text=TITLE, font=_pfont_bold(20),
                 fg=BANNER_FG, bg=BANNER_BG).pack(anchor="w")
        tk.Label(col, text=">> Upload PDF, predict award probability <<",
                 font=_pfont(10), fg="#3d3d00", bg=BANNER_BG).pack(anchor="w")

        # Pixel decoration right side
        tk.Label(inner, text="[v2.0]", font=_pfont(10),
                 fg="#3d3d00", bg=BANNER_BG).pack(side="right")

    # ── ASCII divider ──────────────────────────────────────────────────
    def _ascii_divider(self, ch="═"):
        d = tk.Label(self.wrap, text=ch * 60, font=_pfont(8),
                     fg=CARD_BORDER, bg=BG, anchor="w")
        d.pack(fill="x", pady=(6, 6))

    # ── pixel card factory ─────────────────────────────────────────────
    def _card(self, parent, border_color=CARD_BORDER):
        """Create a card with pixel-style border."""
        outer = tk.Frame(parent, bg=border_color)
        inner = tk.Frame(outer, bg=CARD_BG)
        inner.pack(fill="both", expand=True, padx=2, pady=2)
        return outer, inner

    # ── input card ─────────────────────────────────────────────────────
    def _input_card(self):
        card_o, card = self._card(self.wrap)
        card_o.pack(fill="x", pady=(0, 8))
        body = tk.Frame(card, bg=CARD_BG)
        body.pack(fill="x", padx=20, pady=16)

        # Section header
        tk.Label(body, text="▶ FILE INPUT", font=_pfont_bold(12),
                 fg=ACCENT, bg=CARD_BG).pack(anchor="w", pady=(0, 10))

        # Drop zone
        dz = tk.Frame(body, bg="#0a1628",
                       highlightbackground=ACCENT, highlightthickness=2,
                       cursor="hand2")
        dz.pack(fill="x", ipady=18)

        # Pixel folder icon
        self._drop_icon = tk.Label(dz, text="[  📁  ]",
            font=_pfont(16), fg=ACCENT, bg="#0a1628", cursor="hand2")
        self._drop_icon.pack(pady=(8, 0))

        self._drop_msg = tk.Label(dz,
            text="[ CLICK TO SELECT PDF  or  DRAG & DROP ]",
            font=_pfont(10), fg=TEXT_SEC, bg="#0a1628", cursor="hand2")
        self._drop_msg.pack(pady=(4, 0))

        self.file_label = tk.Label(dz, text="",
            font=_pfont(9), fg=YELLOW, bg="#0a1628")
        self.file_label.pack(pady=(2, 10))

        for w in (dz, self._drop_icon, self._drop_msg, self.file_label):
            w.bind("<Button-1>", lambda e: self._browse())

        if DND_AVAILABLE:
            dz.drop_target_register(DND_FILES)
            dz.dnd_bind("<<Drop>>", self._on_drop)

        self.pdf_path_var.trace_add("write", self._path_changed)

        # Options row
        orow = tk.Frame(body, bg=CARD_BG)
        orow.pack(fill="x", pady=(14, 0))

        tk.Label(orow, text="PROB:", font=_pfont_bold(10),
                 fg=TEXT_SEC, bg=CARD_BG).pack(side="left")
        ttk.Combobox(orow, textvariable=self.problem_var,
                     values=PROBLEMS, width=10, state="readonly",
                     style="P.TCombobox",
                     font=_pfont(10)).pack(side="left", padx=(6, 20))

        tk.Label(orow, text="YEAR:", font=_pfont_bold(10),
                 fg=TEXT_SEC, bg=CARD_BG).pack(side="left")
        tk.Entry(orow, textvariable=self.year_var,
                 font=_pfont(10), width=7, relief="solid",
                 bd=2, bg="#0f3460", fg=TEXT_PRI,
                 insertbackground=ACCENT,
                 highlightbackground=CARD_BORDER,
                 highlightthickness=0).pack(side="left", padx=(6, 0))
        tk.Label(orow, text="(auto if empty)",
                 font=_pfont(8), fg=TEXT_TER, bg=CARD_BG).pack(side="left", padx=(8, 0))

    # ── action bar ─────────────────────────────────────────────────────
    def _action_bar(self):
        bar = tk.Frame(self.wrap, bg=BG)
        bar.pack(fill="x", pady=(6, 6))

        # Pixel-style button with border effect
        btn_border = tk.Frame(bar, bg="#008b8b")
        btn_border.pack(side="left")

        self.btn = tk.Button(btn_border, text="►  RUN PREDICTION  ◄",
            font=_pfont_bold(12),
            bg=ACCENT, fg="#0a0a23",
            activebackground=ACCENT_HOVER, activeforeground="#0a0a23",
            relief="flat", bd=0, cursor="hand2",
            padx=32, pady=8, command=self._predict)
        self.btn.pack(padx=2, pady=2)

        self.status = tk.StringVar(value="")
        self.status_lbl = tk.Label(bar, textvariable=self.status,
            font=_pfont(9), fg=TEXT_SEC, bg=BG)
        self.status_lbl.pack(side="left", padx=(16, 0))

    # ── result placeholder ─────────────────────────────────────────────
    def _result_frame(self):
        self.res = tk.Frame(self.wrap, bg=BG)
        self.res.pack(fill="x")

    # ── callbacks ──────────────────────────────────────────────────────
    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF", "*.pdf"), ("All", "*.*")])
        if p:
            self.pdf_path_var.set(p)

    def _on_drop(self, event):
        p = event.data
        if p.startswith("{") and p.endswith("}"):
            p = p[1:-1]
        self.pdf_path_var.set(p)

    def _path_changed(self, *_):
        p = self.pdf_path_var.get().strip()
        name = os.path.basename(p) if p else ""
        self.file_label.configure(text=f">> {name}" if name else "")
        if p:
            self._drop_msg.configure(text="[ FILE SELECTED ]", fg=GREEN)
        else:
            self._drop_msg.configure(text="[ CLICK TO SELECT PDF  or  DRAG & DROP ]", fg=TEXT_SEC)

    def _sts(self, txt, color=TEXT_SEC):
        self.status.set(txt)
        self.status_lbl.configure(fg=color)

    # ── prediction ─────────────────────────────────────────────────────
    def _predict(self):
        pp = self.pdf_path_var.get().strip().strip('"').strip("'")
        if not pp:
            messagebox.showwarning("ALERT", "请先选择 PDF 文件 / Select PDF first")
            return
        if not os.path.isfile(pp):
            messagebox.showerror("ERROR", f"File not found:\n{pp}")
            return
        prob = self.problem_var.get()
        prob = None if prob == "自动检测" else prob
        ys = self.year_var.get().strip()
        year = int(ys) if ys.isdigit() else None

        self.btn.configure(state="disabled", text="... ANALYZING ...")
        self._sts(">> Loading model & parsing PDF ...", ACCENT)
        threading.Thread(target=self._bg, args=(pp, prob, year), daemon=True).start()

    def _bg(self, pp, prob, year):
        try:
            if self.predictor is None:
                self._tsts(">> First run: loading models ...", YELLOW)
                from predict_award import AwardPredictor
                self.predictor = AwardPredictor()
                self._tsts(">> Models loaded. Analyzing ...", ACCENT)
            r = self.predictor.predict(pp, problem=prob, year=year, verbose=False)
            self.root.after(0, lambda: self._render(r))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._err(str(e)))

    def _tsts(self, t, c=TEXT_SEC):
        self.root.after(0, lambda: self._sts(t, c))

    def _err(self, msg):
        self.btn.configure(state="normal", text="►  RUN PREDICTION  ◄")
        self._sts(f"!! ERROR: {msg}", RED)

    # ==================================================================
    #         ★  RESULT RENDERING  (Pixel-retro cards)
    # ==================================================================
    def _render(self, r):
        self.btn.configure(state="normal", text="►  RUN PREDICTION  ◄")
        if not r.get("success"):
            self._sts(f"!! FAIL: {r.get('error','')}", RED)
            return
        self._sts(">> PREDICTION COMPLETE ✓", GREEN)
        self.result_data = r

        # Clear
        for w in self.res.winfo_children():
            w.destroy()

        probs   = r["probabilities"]
        score   = r["score"]
        sim     = r["similarity"]
        prob    = r["problem"]
        contest = r["contest"]
        year    = r["year"]
        tier    = r["quality_tier"]
        aspects = r.get("aspect_scores", {})
        details = r.get("aspect_details", {})
        meta    = r.get("metadata", {})
        struct  = meta.get("structure", {})

        best_aw = max(probs, key=probs.get)
        best_p  = probs[best_aw]

        # ─────────────── Section header ───────────────────
        tk.Label(self.res, text="╔" + "═" * 50 + "╗",
                 font=_pfont(9), fg=ACCENT, bg=BG).pack(anchor="w")
        tk.Label(self.res, text="║   ★  ANALYSIS RESULTS  ★",
                 font=_pfont_bold(12), fg=ACCENT, bg=BG).pack(anchor="w")
        tk.Label(self.res, text="╚" + "═" * 50 + "╝",
                 font=_pfont(9), fg=ACCENT, bg=BG).pack(anchor="w", pady=(0, 8))

        # ─────────────── 1. Hero score card ───────────────
        hero_o, hero = self._card(self.res, border_color=self._scol(score))
        hero_o.pack(fill="x", pady=(0, 8))
        hi = tk.Frame(hero, bg=CARD_BG)
        hi.pack(fill="x", padx=24, pady=18)

        # Left: big pixel score
        left = tk.Frame(hi, bg=CARD_BG)
        left.pack(side="left", padx=(0, 24))

        tk.Label(left, text="SCORE", font=_pfont_bold(10),
                 fg=TEXT_TER, bg=CARD_BG).pack()

        score_col = self._scol(score)
        tk.Label(left, text=f"{score:.0f}",
                 font=_pfont_bold(48), fg=score_col, bg=CARD_BG).pack()

        tk.Label(left, text="/100", font=_pfont(10),
                 fg=TEXT_TER, bg=CARD_BG).pack()

        # Pixel score bar
        bar_w = 12
        filled = int(score / 100 * bar_w)
        bar_str = PIXEL_CHAR * filled + "░" * (bar_w - filled)
        tk.Label(left, text=f"[{bar_str}]", font=_pfont(10),
                 fg=score_col, bg=CARD_BG).pack(pady=(4, 0))

        # Right: info column
        ri = tk.Frame(hi, bg=CARD_BG)
        ri.pack(side="left", fill="both", expand=True)

        tk.Label(ri, text=f"CONTEST: {contest}  |  PROBLEM: {prob}  |  YEAR: {year}",
                 font=_pfont(10), fg=TEXT_SEC, bg=CARD_BG).pack(anchor="w")

        tk.Label(ri, text=f"TIER:  {tier}",
                 font=_pfont_bold(16), fg=TEXT_PRI, bg=CARD_BG).pack(anchor="w", pady=(8, 4))

        tk.Label(ri, text=f"Cosine Similarity: {sim:.4f}",
                 font=_pfont(9), fg=TEXT_TER, bg=CARD_BG).pack(anchor="w")

        # Award badge (pixel style)
        aw_full = {"O": "Outstanding Winner", "F": "Finalist",
                   "M": "Meritorious Winner", "H": "Honorable Mention",
                   "S": "Successful Participant"}
        bcol = GREEN if best_aw in "OF" else (ORANGE if best_aw == "M" else TEXT_SEC)
        badge_text = f"▓▓  {best_aw} - {aw_full.get(best_aw,'')}  [{best_p*100:.1f}%]  ▓▓"
        badge = tk.Frame(ri, bg="#0a1628")
        badge.pack(anchor="w", pady=(12, 0))
        tk.Label(badge, text=badge_text,
                 font=_pfont_bold(11), fg=bcol, bg="#0a1628").pack(padx=8, pady=4)

        # ─────────────── 2. Three aspect cards ────────────
        tk.Label(self.res, text="── ASPECT BREAKDOWN ──",
                 font=_pfont_bold(10), fg=TEXT_TER, bg=BG).pack(anchor="w", pady=(8, 6))

        trio = tk.Frame(self.res, bg=BG)
        trio.pack(fill="x", pady=(0, 8))

        a_meta = [
            ("abstract", "ABSTRACT",  "摘要写作"),
            ("figures",  "FIGURES",   "图表水平"),
            ("modeling", "MODELING",  "建模深度"),
        ]

        for i, (key, en_lbl, cn_lbl) in enumerate(a_meta):
            s  = aspects.get(key, 0)
            sc = self._scol(s)
            dt = details.get(key, {})
            sv = dt.get("similarity", 0)
            diffs = dt.get("differences", [])

            c_o, c = self._card(trio, border_color=sc)
            c_o.pack(side="left", fill="both", expand=True,
                     padx=(0 if i == 0 else 6, 0))

            inn = tk.Frame(c, bg=CARD_BG)
            inn.pack(fill="x", padx=14, pady=14)

            # Header
            tk.Label(inn, text=f"▸ {en_lbl}", font=_pfont_bold(10),
                     fg=ACCENT, bg=CARD_BG).pack(anchor="w")
            tk.Label(inn, text=cn_lbl, font=_pfont(9),
                     fg=TEXT_TER, bg=CARD_BG).pack(anchor="w")

            # Big score
            sr = tk.Frame(inn, bg=CARD_BG)
            sr.pack(anchor="w", pady=(6, 4))
            tk.Label(sr, text=f"{s:.0f}",
                     font=_pfont_bold(28), fg=sc, bg=CARD_BG).pack(side="left")
            tk.Label(sr, text=" /100",
                     font=_pfont(9), fg=TEXT_TER,
                     bg=CARD_BG).pack(side="left", pady=(12, 0))

            # Pixel bar
            bw = 10
            bf = int(s / 100 * bw)
            bs = PIXEL_CHAR * bf + "░" * (bw - bf)
            tk.Label(inn, text=f"[{bs}]", font=_pfont(9),
                     fg=sc, bg=CARD_BG).pack(anchor="w")

            # Sim value
            tk.Label(inn, text=f"sim: {sv:.3f}",
                     font=_pfont(8), fg=TEXT_TER,
                     bg=CARD_BG).pack(anchor="w", pady=(6, 2))

            # Diff items
            for d in diffs[:3]:
                fg = (GREEN if any(k in d for k in ("合理","充足"))
                      else RED if any(k in d for k in ("偏少","偏短","偏低","缺少","缺乏","不足","未检测"))
                      else TEXT_SEC)
                prefix = "+" if fg == GREEN else ("-" if fg == RED else "·")
                tk.Label(inn, text=f" {prefix} {d}", font=_pfont(8),
                         fg=fg, bg=CARD_BG, wraplength=200,
                         justify="left").pack(anchor="w")

        # ─────────────── 3. Probability distribution ──────
        tk.Label(self.res, text="── AWARD PROBABILITY ──",
                 font=_pfont_bold(10), fg=TEXT_TER, bg=BG).pack(anchor="w", pady=(8, 6))

        pc_o, pc = self._card(self.res)
        pc_o.pack(fill="x", pady=(0, 8))
        pi = tk.Frame(pc, bg=CARD_BG)
        pi.pack(fill="x", padx=24, pady=18)

        tk.Label(pi, text="PROBABILITY DISTRIBUTION",
                 font=_pfont_bold(12), fg=ACCENT, bg=CARD_BG).pack(anchor="w")
        tk.Label(pi, text=f"Based on Problem {prob} historical priors",
                 font=_pfont(8), fg=TEXT_TER,
                 bg=CARD_BG).pack(anchor="w", pady=(2, 12))

        aw_short = {"O": "Outstanding", "F": "Finalist",
                    "M": "Meritorious", "H": "Honorable",
                    "S": "Successful "}
        aw_icons = {"O": "♛", "F": "♚", "M": "♜", "H": "♞", "S": "♟"}
        mx = max(probs.values()) if probs else 1

        for aw in ["O", "F", "M", "H", "S"]:
            p = probs.get(aw, 0)
            is_best = (aw == best_aw)
            row = tk.Frame(pi, bg=CARD_BG)
            row.pack(fill="x", pady=3)

            # Icon + letter
            icn = aw_icons.get(aw, "?")
            aw_col = ACCENT if is_best else TEXT_SEC
            tk.Label(row, text=f"{icn} {aw}",
                     font=_pfont_bold(11), fg=aw_col,
                     bg=CARD_BG, width=4, anchor="w").pack(side="left")

            tk.Label(row, text=aw_short[aw],
                     font=_pfont(9), fg=TEXT_TER,
                     bg=CARD_BG, width=13, anchor="w").pack(side="left")

            # Pixel bar
            max_blocks = 20
            n_blocks = int(p / max(mx, 0.01) * max_blocks * 0.9) if p > 0 else 0
            n_blocks = max(n_blocks, 1) if p > 0.001 else 0
            empty_blocks = max_blocks - n_blocks

            bar_col = GREEN if is_best else TEXT_TER
            bar_text = PIXEL_CHAR * n_blocks + "░" * empty_blocks
            tk.Label(row, text=bar_text, font=_pfont(9),
                     fg=bar_col, bg=CARD_BG, anchor="w").pack(side="left", padx=(4, 8))

            # Percentage
            pcol = ACCENT if is_best else TEXT_SEC
            pct = f"{p*100:.1f}%"
            tk.Label(row, text=pct, font=_pfont_bold(10),
                     fg=pcol, bg=CARD_BG, width=7,
                     anchor="e").pack(side="right")

            # Star marker for best
            if is_best:
                tk.Label(row, text="◄ BEST", font=_pfont(8),
                         fg=YELLOW, bg=CARD_BG).pack(side="right", padx=(0, 4))

        desc = r.get("description", "")
        if desc:
            tk.Label(pi, text="─" * 40, font=_pfont(8),
                     fg=CARD_BORDER, bg=CARD_BG).pack(anchor="w", pady=(10, 4))
            tk.Label(pi, text=desc.replace("\n", "  ").strip(),
                     font=_pfont(9), fg=TEXT_SEC, bg=CARD_BG,
                     wraplength=620, justify="left").pack(anchor="w")

        # ─────────────── 4. Paper metadata card ───────────
        tk.Label(self.res, text="── PAPER  METADATA ──",
                 font=_pfont_bold(10), fg=TEXT_TER, bg=BG).pack(anchor="w", pady=(8, 6))

        mc_o, mc = self._card(self.res)
        mc_o.pack(fill="x", pady=(0, 8))
        mi = tk.Frame(mc, bg=CARD_BG)
        mi.pack(fill="x", padx=24, pady=18)

        tk.Label(mi, text="DOCUMENT INFO", font=_pfont_bold(12),
                 fg=ACCENT, bg=CARD_BG).pack(anchor="w", pady=(0, 10))

        rows = [
            ("PAGES",    f"{meta.get('page_count',0)}"),
            ("IMAGES",   f"{meta.get('image_count',0)} imgs  |  "
                         f"{struct.get('figure_caption_count',0)} captions  |  "
                         f"{struct.get('table_count',0)} tables"),
            ("FORMULAS", f"{struct.get('formula_count',0)}"),
            ("REFS",     f"{struct.get('citation_count',0)} citations  |  "
                         f"{meta.get('ref_count',0)} references"),
            ("ABSTRACT", f"{meta.get('abstract_length',0)} chars"),
            ("CONTENT",  f"~{meta.get('full_text_length',0)//1000}k chars"),
        ]
        for lb, vl in rows:
            rr = tk.Frame(mi, bg=CARD_BG)
            rr.pack(fill="x", pady=2)
            tk.Label(rr, text=f"  {lb}:", font=_pfont_bold(9), fg=TEXT_SEC,
                     bg=CARD_BG, width=12, anchor="w").pack(side="left")
            tk.Label(rr, text=vl, font=_pfont(9), fg=TEXT_PRI,
                     bg=CARD_BG, anchor="w").pack(side="left")

        # Separator
        tk.Label(mi, text="─" * 40, font=_pfont(8),
                 fg=CARD_BORDER, bg=CARD_BG).pack(anchor="w", pady=(10, 8))

        # Completeness
        comp = struct.get("structure_completeness", 0)
        comp_blocks = int(comp * 15)
        comp_bar = PIXEL_CHAR * comp_blocks + "░" * (15 - comp_blocks)
        comp_col = self._scol(comp * 100)
        tk.Label(mi, text=f"COMPLETENESS: [{comp_bar}] {comp:.0%}",
                 font=_pfont(9), fg=comp_col, bg=CARD_BG).pack(anchor="w")

        # Tags: section flags
        tk.Label(mi, text="", bg=CARD_BG).pack(pady=(4, 0))  # spacer
        flags = [
            ("ABSTRACT",  struct.get("has_abstract", False)),
            ("INTRO",     struct.get("has_introduction", False)),
            ("METHOD",    struct.get("has_methodology", False)),
            ("RESULTS",   struct.get("has_results", False)),
            ("CONCLUDE",  struct.get("has_conclusion", False)),
            ("REFS",      struct.get("has_references", False)),
        ]
        trow = tk.Frame(mi, bg=CARD_BG)
        trow.pack(anchor="w")
        for nm, ok in flags:
            sym = "■" if ok else "□"
            fg2 = GREEN if ok else RED
            tk.Label(trow, text=f"{sym} {nm}", font=_pfont(8),
                     fg=fg2, bg=CARD_BG).pack(side="left", padx=(0, 8))

        # Advanced features
        adv = []
        if struct.get("has_sensitivity_analysis"): adv.append("SENSITIVITY")
        if struct.get("has_model_validation"):     adv.append("VALIDATION")
        if struct.get("has_strengths_weaknesses"): adv.append("S/W ANALYSIS")
        if struct.get("has_future_work"):          adv.append("FUTURE WORK")

        if adv:
            ar = tk.Frame(mi, bg=CARD_BG)
            ar.pack(anchor="w", pady=(8, 0))
            tk.Label(ar, text="BONUS:", font=_pfont_bold(8),
                     fg=ACCENT, bg=CARD_BG).pack(side="left", padx=(0, 6))
            for a in adv:
                tk.Label(ar, text=f"[{a}]", font=_pfont(8),
                         fg=ACCENT, bg=ACCENT_LIGHT).pack(side="left", padx=(0, 4))
        else:
            tk.Label(mi, text="TIP: Add sensitivity analysis, validation, S/W analysis",
                     font=_pfont(8), fg=ORANGE,
                     bg=CARD_BG).pack(anchor="w", pady=(8, 0))

        # Footer
        tk.Label(self.res, text="═" * 60, font=_pfont(8),
                 fg=CARD_BORDER, bg=BG).pack(anchor="w", pady=(6, 0))
        tk.Label(self.res, text=">> END OF ANALYSIS REPORT <<",
                 font=_pfont(9), fg=TEXT_TER, bg=BG).pack(anchor="w", pady=(2, 24))

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _scol(s):
        if s >= 80: return GREEN
        if s >= 60: return ORANGE
        return RED

    # ── run ────────────────────────────────────────────────────────────
    def run(self):
        self.root.mainloop()


def main():
    PredictorGUI().run()


if __name__ == "__main__":
    main()
