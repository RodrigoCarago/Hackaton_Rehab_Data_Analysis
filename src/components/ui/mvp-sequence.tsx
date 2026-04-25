"use client";

import { useMemo, useState, type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload, Check, Zap, Loader2,
  AlertCircle, Activity, TrendingUp, TrendingDown, Minus,
} from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────
type FileRole = "pre-train" | "pre-test" | "post-train" | "post-test";
interface UploadedFile { role: FileRole; file: File }
interface Filters { highPassHz: number; lowPassHz: number }
interface HyperParams {
  epochTmin: number;
  epochTmax: number;
  cspComponents: number;
  fbcspKBest: number;
}
type TemporalViewMode = "raw" | "filtered" | "both";
type DetailTab = "pre" | "post";

interface ApiMetric {
  accuracy: number; kappa: number; f1: number;
  precision: number; recall: number; roc_auc: number;
}
interface ApiResult {
  fs: number;
  n_epochs: { train: number; test: number };
  channel_names: string[];
  temporal: {
    raw: number[][];
    filtered: number[][];
    n_samples: number;
    duration_s: number;
    events?: { t: number; label: string }[];
  };
  temporal_test?: {
    raw: number[][];
    filtered: number[][];
    n_samples: number;
    duration_s: number;
    events?: { t: number; label: string }[];
  };
  psd: { freqs: number[]; train: number[]; test: number[] };
  metrics: {
    csp_lda: ApiMetric;
    fbcsp_lda: ApiMetric;
    evaluation?: { method: string; folds: number; dataset?: string };
  };
  confusion: { csp_lda: number[][]; fbcsp_lda: number[][] };
  roc: { csp_lda: { fpr: number[]; tpr: number[] }; fbcsp_lda: { fpr: number[]; tpr: number[] } };
}

// ─── Constants ────────────────────────────────────────────────────────────────
const FALLBACK_CH = ["FC3","FCz","FC4","C5","C3","C1","Cz","C2","C4","C6","CP3","CP1","CPz","CP2","CP4","Pz"] as const;
const FIXED_CHANNELS: string[] = [...FALLBACK_CH];
const CH_COLORS = ["#60a5fa","#34d399","#a78bfa","#fb923c","#f472b6","#facc15","#22d3ee","#f87171"];
const SVG_W = 900;
const T_ML = 48; const T_MT = 10; const T_LANE = 46;
const API = "http://localhost:8000";

// ─── Utilities ────────────────────────────────────────────────────────────────
function channelColor(i: number) {
  if (i < CH_COLORS.length) return CH_COLORS[i];
  return `hsl(${(i * 47) % 360} 85% 68%)`;
}
function fmtBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 ** 2).toFixed(2)} MB`;
}
function svgPath(pts: number[], x0: number, w: number, y0: number, h: number): string {
  return pts.map((v, i) => {
    const x = x0 + (i / (pts.length - 1)) * w;
    const y = y0 + h / 2 - v * h * 0.42;
    return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
}
function scaleSeriesForLane(series: number[]): number[] {
  if (!series.length) return series;
  let min = Infinity;
  let max = -Infinity;
  for (const v of series) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = Math.max(max - min, 1e-9);
  const center = (max + min) / 2;
  return series.map((v) => ((v - center) / range) * 2.0);
}
function perfTier(acc: number): { label: string; cls: string } {
  if (acc >= 85) return { label: "Excellent", cls: "bg-emerald-500/10 text-emerald-300 border-emerald-500/20" };
  if (acc >= 75) return { label: "Good",      cls: "bg-blue-500/10 text-blue-300 border-blue-500/20"         };
  if (acc >= 65) return { label: "Fair",      cls: "bg-amber-500/10 text-amber-400 border-amber-500/20"      };
  if (acc >= 55) return { label: "Below Avg", cls: "bg-orange-500/10 text-orange-400 border-orange-500/20"   };
  return               { label: "Chance",    cls: "bg-red-500/10 text-red-400 border-red-500/20"             };
}

// ─── Section Label ────────────────────────────────────────────────────────────
function SectionLabel({ children, accent = "indigo" }: { children: ReactNode; accent?: "indigo" | "emerald" }) {
  return (
    <span className={cn(
      "text-[10px] font-semibold uppercase tracking-[0.15em]",
      accent === "emerald" ? "text-emerald-400/70" : "text-slate-400",
    )}>
      {children}
    </span>
  );
}

// ─── DropZone ─────────────────────────────────────────────────────────────────
function DropZone({ role, label, file, onFile }: {
  role: FileRole; label: string; file: UploadedFile | null;
  onFile: (role: FileRole, f: File) => void;
}) {
  const [drag, setDrag] = useState(false);
  const loaded = Boolean(file);
  return (
    <label className={cn(
      "group relative flex cursor-pointer items-center gap-3 rounded border px-3 py-2.5 transition-all duration-200",
      drag   ? "border-indigo-400/40 bg-indigo-500/[0.05]" :
      loaded ? "border-emerald-500/25 bg-emerald-500/[0.03]" :
               "border-white/[0.06] bg-white/[0.015] hover:border-white/[0.10] hover:bg-white/[0.025]",
    )}
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files?.[0]; if (f) onFile(role, f); }}
    >
      <input type="file" accept=".mat,.csv,.edf" className="sr-only"
        onChange={e => { const f = e.target.files?.[0]; if (f) onFile(role, f); }} />

      <div className={cn(
        "flex h-7 w-7 shrink-0 items-center justify-center rounded-sm border",
        loaded ? "border-emerald-500/30 bg-emerald-500/10" : "border-white/[0.07] bg-white/[0.03]",
      )}>
        {loaded
          ? <Check className="h-3.5 w-3.5 text-emerald-400" />
          : <Upload className="h-3 w-3 text-slate-600 group-hover:text-slate-500" />}
      </div>

      <div className="min-w-0 flex-1">
        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">{label}</p>
        {loaded ? (
          <p className="mt-0.5 truncate font-mono text-[11px] text-slate-300">{file!.file.name}</p>
        ) : (
          <p className="mt-0.5 text-[10px] text-slate-600 group-hover:text-slate-500">
            Drop or click · .mat .csv .edf
          </p>
        )}
      </div>

      {loaded && (
        <span className="shrink-0 font-mono text-[9px] text-slate-500">{fmtBytes(file!.file.size)}</span>
      )}
    </label>
  );
}

// ─── Slider ───────────────────────────────────────────────────────────────────
function Slider({ label, value, min, max, unit, onChange, step = 1 }: {
  label: string; value: number; min: number; max: number; unit: string;
  onChange: (v: number) => void; step?: number;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">{label}</span>
        <span className="font-mono text-[11px] tabular-nums text-slate-300">{value}{unit && ` ${unit}`}</span>
      </div>
      <div className="relative h-px rounded-full bg-white/[0.08]">
        <div className="absolute inset-y-0 left-0 rounded-full bg-indigo-500/40 transition-all" style={{ width: `${pct}%` }} />
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="absolute inset-0 h-4 w-full -translate-y-1.5 cursor-pointer opacity-0" />
      </div>
    </div>
  );
}

// ─── Metrics Card ─────────────────────────────────────────────────────────────
function MetricsCard({ m, label, accentColor }: { m: ApiMetric; label: string; accentColor: string }) {
  const tier = perfTier(m.accuracy);
  return (
    <div className="rounded border border-white/[0.06] bg-[#0a0c14]">
      <div className="flex items-center justify-between border-b border-white/[0.04] px-4 py-2.5">
        <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-slate-400">{label}</span>
        <span className={cn("rounded-sm border px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider", tier.cls)}>
          {tier.label}
        </span>
      </div>
      <div className="px-4 pt-3 pb-4">
        <div className="mb-0.5 flex items-baseline gap-1">
          <span className="text-[2.25rem] font-light tabular-nums leading-none text-slate-50">{m.accuracy.toFixed(2)}</span>
          <span className="text-sm text-slate-600">%</span>
        </div>
        <div className="mb-4 h-[2px] w-full overflow-hidden rounded-full bg-white/[0.05]">
          <motion.div
            className="h-full rounded-full"
            style={{ background: accentColor }}
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(100, m.accuracy)}%` }}
            transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
          />
        </div>
        <div className="grid grid-cols-5 gap-x-2 gap-y-2.5 border-t border-white/[0.04] pt-3">
          {([
            ["κ",         m.kappa.toFixed(3)],
            ["ROC AUC",   m.roc_auc.toFixed(3)],
            ["F1",        m.f1.toFixed(3)],
            ["Prec",      m.precision.toFixed(3)],
            ["Recall",    m.recall.toFixed(3)],
          ] as const).map(([name, val]) => (
            <div key={name} className="text-center">
              <p className="font-mono text-[11px] tabular-nums text-slate-200">{val}</p>
              <p className="mt-0.5 text-[9px] uppercase tracking-widest text-slate-600">{name}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Delta Chip ───────────────────────────────────────────────────────────────
function DeltaChip({ delta, format = "dec" }: { delta: number; format?: "pct" | "dec" }) {
  const pos = delta > 0.0005;
  const neg = delta < -0.0005;
  const str = format === "pct"
    ? `${pos ? "+" : ""}${delta.toFixed(1)}%`
    : `${pos ? "+" : ""}${delta.toFixed(3)}`;
  return (
    <span className={cn(
      "inline-flex items-center gap-0.5 rounded-sm px-1.5 py-0.5 text-[10px] font-bold font-mono tabular-nums",
      pos ? "bg-emerald-500/10 text-emerald-300" :
      neg ? "bg-red-500/10 text-red-400" :
            "bg-white/[0.04] text-slate-600",
    )}>
      {pos ? <TrendingUp className="h-2.5 w-2.5" /> : neg ? <TrendingDown className="h-2.5 w-2.5" /> : <Minus className="h-2.5 w-2.5" />}
      {str}
    </span>
  );
}

// ─── Comparison Panel ─────────────────────────────────────────────────────────
function ComparisonPanel({ pre, post }: { pre: ApiResult; post: ApiResult }) {
  const rows: { label: string; key: keyof ApiMetric; fmt: "pct" | "dec" }[] = [
    { label: "Accuracy",  key: "accuracy",  fmt: "pct" },
    { label: "κ Kappa",   key: "kappa",     fmt: "dec" },
    { label: "ROC AUC",   key: "roc_auc",   fmt: "dec" },
    { label: "F1 Macro",  key: "f1",        fmt: "dec" },
    { label: "Precision", key: "precision", fmt: "dec" },
    { label: "Recall",    key: "recall",    fmt: "dec" },
  ];

  function fmt(v: number, format: "pct" | "dec") {
    return format === "pct" ? `${v.toFixed(2)}%` : v.toFixed(4);
  }

  const pipes = [
    { key: "csp_lda" as const,   label: "CSP + LDA",   accent: "rgba(99,102,241,0.70)" },
    { key: "fbcsp_lda" as const, label: "FBCSP + LDA", accent: "rgba(251,146,60,0.70)"  },
  ];

  return (
    <div className="space-y-4">
      {pipes.map(pipe => {
        const mPre  = pre.metrics[pipe.key];
        const mPost = post.metrics[pipe.key];
        const preTier  = perfTier(mPre.accuracy);
        const postTier = perfTier(mPost.accuracy);

        return (
          <div key={pipe.key} className="overflow-hidden rounded border border-white/[0.06] bg-[#0a0c14]">
            {/* Header */}
            <div className="flex items-center gap-2.5 border-b border-white/[0.05] px-4 py-2.5">
              <span className="inline-block h-[3px] w-5 rounded-full" style={{ background: pipe.accent }} />
              <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400">{pipe.label}</span>
            </div>

            {/* Accuracy comparison */}
            <div className="grid grid-cols-[1fr_72px_1fr] gap-3 px-4 py-4">
              <div>
                <div className="mb-1.5 flex items-center justify-between">
                  <SectionLabel>Pre-Rehab</SectionLabel>
                  <span className={cn("rounded-sm border px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider", preTier.cls)}>{preTier.label}</span>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-light tabular-nums leading-none text-slate-100">{mPre.accuracy.toFixed(1)}</span>
                  <span className="text-xs text-slate-600">%</span>
                </div>
                <div className="mt-2 h-[2px] w-full overflow-hidden rounded-full bg-white/[0.05]">
                  <motion.div className="h-full rounded-full bg-indigo-500/40"
                    initial={{ width: 0 }} animate={{ width: `${Math.min(100, mPre.accuracy)}%` }}
                    transition={{ duration: 0.9, ease: "easeOut" }} />
                </div>
              </div>

              <div className="flex items-center justify-center">
                <DeltaChip delta={mPost.accuracy - mPre.accuracy} format="pct" />
              </div>

              <div>
                <div className="mb-1.5 flex items-center justify-between">
                  <SectionLabel accent="emerald">Post-Rehab</SectionLabel>
                  <span className={cn("rounded-sm border px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider", postTier.cls)}>{postTier.label}</span>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-light tabular-nums leading-none text-slate-100">{mPost.accuracy.toFixed(1)}</span>
                  <span className="text-xs text-slate-600">%</span>
                </div>
                <div className="mt-2 h-[2px] w-full overflow-hidden rounded-full bg-white/[0.05]">
                  <motion.div className="h-full rounded-full"
                    style={{ background: pipe.accent }}
                    initial={{ width: 0 }} animate={{ width: `${Math.min(100, mPost.accuracy)}%` }}
                    transition={{ duration: 0.9, ease: "easeOut", delay: 0.1 }} />
                </div>
              </div>
            </div>

            {/* Detailed table */}
            <div className="border-t border-white/[0.04] px-4 pb-3 pt-2.5">
              <table className="w-full text-xs">
                <thead>
                  <tr>
                    <th className="pb-2 text-left text-[10px] font-semibold uppercase tracking-widest text-slate-500">Metric</th>
                    <th className="pb-2 text-right text-[10px] font-semibold uppercase tracking-widest text-slate-500">Pre</th>
                    <th className="pb-2 text-right text-[10px] font-semibold uppercase tracking-widest text-slate-500">Post</th>
                    <th className="pb-2 text-right text-[10px] font-semibold uppercase tracking-widest text-slate-500">Δ</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.03]">
                  {rows.map(({ label, key, fmt: f }) => {
                    const vPre  = mPre[key]  as number;
                    const vPost = mPost[key] as number;
                    return (
                      <tr key={key}>
                        <td className="py-1.5 text-[11px] font-medium text-slate-500">{label}</td>
                        <td className="py-1.5 text-right font-mono text-[11px] tabular-nums text-slate-500">{fmt(vPre, f)}</td>
                        <td className="py-1.5 text-right font-mono text-[11px] tabular-nums text-slate-200">{fmt(vPost, f)}</td>
                        <td className="py-1.5 text-right">
                          <DeltaChip delta={vPost - vPre} format={f} />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── ROC Chart ────────────────────────────────────────────────────────────────
function RocChart({ result }: { result: ApiResult }) {
  const W = 400; const H = 200;
  const ML = 30; const MT = 10; const MB = 24;
  const plotW = W - ML - 10; const plotH = H - MT - MB;

  function rocPath(fpr: number[], tpr: number[]) {
    return fpr.map((f, i) => {
      const x = ML + f * plotW;
      const y = MT + (1 - tpr[i]) * plotH;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(" ");
  }

  return (
    <div className="rounded border border-white/[0.06] bg-[#0a0c14]">
      <svg viewBox={`0 0 ${W} ${H}`} style={{ height: `${H}px`, width: "100%" }}>
        {Array.from({ length: 5 }, (_, i) => (
          <line key={i} x1={ML} y1={MT + (i / 4) * plotH} x2={ML + plotW} y2={MT + (i / 4) * plotH}
            stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
        ))}
        <path d={`M ${ML},${MT + plotH} L ${ML + plotW},${MT}`} fill="none"
          stroke="rgba(255,255,255,0.08)" strokeWidth="1" strokeDasharray="4 4" />
        <motion.path d={rocPath(result.roc.csp_lda.fpr, result.roc.csp_lda.tpr)}
          fill="none" stroke="#60a5fa" strokeWidth="1.8" strokeOpacity="0.85"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
          transition={{ duration: 1.4, ease: "easeOut" }} />
        <motion.path d={rocPath(result.roc.fbcsp_lda.fpr, result.roc.fbcsp_lda.tpr)}
          fill="none" stroke="#fb923c" strokeWidth="1.6" strokeOpacity="0.85"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
          transition={{ duration: 1.5, ease: "easeOut", delay: 0.1 }} />
        <text x={ML + 4} y={MT + 14} fontSize="9" fill="rgba(255,255,255,0.20)" fontFamily="ui-monospace,monospace">TPR</text>
        <text x={ML + plotW / 2} y={H - 4} textAnchor="middle" fontSize="9"
          fill="rgba(255,255,255,0.18)" fontFamily="ui-monospace,monospace">FPR</text>
        <g transform={`translate(${ML + 8},${MT + plotH - 32})`}>
          <line x1="0" y1="6"  x2="10" y2="6"  stroke="#60a5fa" strokeWidth="1.8" />
          <text x="13" y="10" fontSize="8" fill="rgba(255,255,255,0.35)" fontFamily="ui-monospace,monospace">
            CSP · AUC {result.metrics.csp_lda.roc_auc.toFixed(3)}
          </text>
          <line x1="0" y1="19" x2="10" y2="19" stroke="#fb923c" strokeWidth="1.6" />
          <text x="13" y="23" fontSize="8" fill="rgba(255,255,255,0.35)" fontFamily="ui-monospace,monospace">
            FBCSP · AUC {result.metrics.fbcsp_lda.roc_auc.toFixed(3)}
          </text>
        </g>
      </svg>
    </div>
  );
}

// ─── Temporal Chart ───────────────────────────────────────────────────────────
function TemporalChart({ windowSeconds, apiData, visibleChannels, viewMode, viewportStartSec = 0, viewportDurationSec }: {
  windowSeconds: number;
  apiData: (ApiResult["temporal"] & { channel_names: string[] }) | null;
  visibleChannels: string[];
  viewMode: TemporalViewMode;
  viewportStartSec?: number;
  viewportDurationSec?: number;
}) {
  const viewportDuration = viewportDurationSec ?? windowSeconds;
  const baseNames = useMemo(() => apiData?.channel_names ?? [...FALLBACK_CH], [apiData]);
  const selectedNames = useMemo(() => {
    const s = baseNames.filter(ch => visibleChannels.includes(ch));
    return s.length ? s : [baseNames[0]];
  }, [baseNames, visibleChannels]);
  const selectedIdx = selectedNames.map(ch => baseNames.indexOf(ch)).filter(i => i >= 0);

  const empty = useMemo(() => baseNames.map(() => Array.from({ length: 240 }, () => 0)), [baseNames]);
  const rawBase  = apiData ? apiData.raw      : empty;
  const filtBase = apiData ? apiData.filtered : empty;
  const totalSamples = rawBase[0]?.length ?? 0;
  const totalDuration = apiData?.duration_s ?? windowSeconds;
  const safeDuration = Math.max(totalDuration, 1e-6);
  const viewStartClamped = Math.max(0, Math.min(viewportStartSec, safeDuration));
  const viewDurClamped = Math.max(Math.min(viewportDuration, safeDuration), safeDuration / Math.max(totalSamples, 1));
  const viewEndClamped = Math.min(safeDuration, viewStartClamped + viewDurClamped);
  const startIdx = Math.max(0, Math.floor((viewStartClamped / safeDuration) * totalSamples));
  const endIdx = Math.max(startIdx + 2, Math.min(totalSamples, Math.ceil((viewEndClamped / safeDuration) * totalSamples)));

  const rawBaseView = useMemo(
    () => rawBase.map((s) => s.slice(startIdx, endIdx)),
    [rawBase, startIdx, endIdx],
  );
  const filtBaseView = useMemo(
    () => filtBase.map((s) => s.slice(startIdx, endIdx)),
    [filtBase, startIdx, endIdx],
  );

  const rawSigs = useMemo(
    () => selectedIdx.map((i) => rawBaseView[i]).filter(Boolean),
    [selectedIdx, rawBaseView],
  );
  const filtSigs = useMemo(
    () => selectedIdx.map((i) => filtBaseView[i]).filter(Boolean),
    [selectedIdx, filtBaseView],
  );
  const rawDisplay = useMemo(
    () => rawSigs.map(scaleSeriesForLane),
    [rawSigs],
  );
  const filtDisplay = useMemo(
    () => filtSigs.map(scaleSeriesForLane),
    [filtSigs],
  );

  const nCh    = Math.max(1, selectedNames.length);
  const totalH = nCh * T_LANE + T_MT + 22;
  const plotW  = SVG_W - T_ML - 16;

  const rawPaths  = useMemo(() => rawDisplay.map( (s, i) => svgPath(s, T_ML, plotW, T_MT + i * T_LANE, T_LANE)), [rawDisplay,  plotW]);
  const filtPaths = useMemo(() => filtDisplay.map((s, i) => svgPath(s, T_ML, plotW, T_MT + i * T_LANE, T_LANE)), [filtDisplay, plotW]);

  const duration = `${viewStartClamped.toFixed(1)}s – ${viewEndClamped.toFixed(1)}s`;

  return (
    <div className="overflow-x-auto rounded border border-white/[0.06] bg-[#0a0c14]">
      <svg viewBox={`0 0 ${SVG_W} ${totalH}`} style={{ height: `${totalH}px`, width: "100%", minWidth: "680px" }}>
        {Array.from({ length: 11 }, (_, i) => (
          <line key={i} x1={T_ML + (i / 10) * plotW} y1={T_MT} x2={T_ML + (i / 10) * plotW} y2={totalH - 22}
            stroke="rgba(255,255,255,0.025)" strokeWidth="1" />
        ))}
        {selectedNames.map((_, i) => (
          <line key={`sep-${i}`} x1={T_ML} y1={T_MT + (i + 1) * T_LANE} x2={T_ML + plotW} y2={T_MT + (i + 1) * T_LANE}
            stroke="rgba(255,255,255,0.02)" strokeWidth="1" />
        ))}
        {selectedNames.map((ch, i) => (
          <text key={`lbl-${ch}`} x={T_ML - 5} y={T_MT + i * T_LANE + T_LANE / 2 + 4}
            textAnchor="end" fontSize="11" fontFamily="ui-monospace,monospace" fill={channelColor(i)}>{ch}</text>
        ))}
        {(viewMode === "raw" || viewMode === "both") && rawPaths.map((d, i) => (
          <motion.path key={`r${i}`} d={d} fill="none"
            stroke={channelColor(i)} strokeWidth="1.1" strokeOpacity="0.25"
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
            transition={{ duration: 1.4, ease: "easeInOut", delay: i * 0.05 }} />
        ))}
        {(viewMode === "filtered" || viewMode === "both") && filtPaths.map((d, i) => (
          <motion.path key={`f${i}`} d={d} fill="none"
            stroke={channelColor(i)} strokeWidth="1.9" strokeOpacity="0.88"
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
            transition={{ duration: 1.6, ease: "easeInOut", delay: i * 0.06 + 0.12 }} />
        ))}
        {(apiData?.events ?? []).map((ev, i) => {
          if (ev.t < viewStartClamped || ev.t > viewEndClamped) return null;
          const x = T_ML + ((ev.t - viewStartClamped) / Math.max(viewEndClamped - viewStartClamped, 1e-6)) * plotW;
          const isLeft = ev.label === "left";
          return (
            <g key={`ev-${i}-${ev.t}`}>
              <line x1={x} y1={T_MT} x2={x} y2={totalH - 22}
                stroke={isLeft ? "rgba(52,211,153,0.40)" : "rgba(239,68,68,0.40)"}
                strokeWidth="1.2" strokeDasharray="3 3" />
              <circle cx={x} cy={T_MT + 4} r={2}
                fill={isLeft ? "rgba(52,211,153,0.90)" : "rgba(239,68,68,0.90)"} />
            </g>
          );
        })}
        {Array.from({ length: 6 }, (_, i) => (
          <text key={`t${i}`} x={T_ML + (i / 5) * plotW} y={totalH - 5}
            textAnchor="middle" fontSize="9" fontFamily="ui-monospace,monospace"
            fill="rgba(255,255,255,0.18)">
            {(viewStartClamped + (i / 5) * (viewEndClamped - viewStartClamped)).toFixed(1)}s
          </text>
        ))}
        <text x={T_ML + plotW - 2} y={T_MT + 10} textAnchor="end" fontSize="8"
          fill="rgba(255,255,255,0.15)" fontFamily="ui-monospace,monospace">{duration}</text>
        {!apiData && (
          <text x={T_ML + plotW / 2} y={T_MT + (totalH - 22) / 2}
            textAnchor="middle" fontSize="12" fontFamily="ui-monospace,monospace"
            fill="rgba(255,255,255,0.18)">Run processing to see EEG signal</text>
        )}
      </svg>
    </div>
  );
}

// ─── Session Detail ───────────────────────────────────────────────────────────
function SessionDetail({ result, phase, epochWindow }: {
  result: ApiResult;
  phase: "pre" | "post";
  epochWindow: number;
}) {
  const [viewMode, setViewMode] = useState<TemporalViewMode>("raw");
  const [zoomX, setZoomX] = useState<number>(8);
  const [panNorm, setPanNorm] = useState<number>(0);
  const accentColor = phase === "pre" ? "rgba(99,102,241,0.70)" : "rgba(52,211,153,0.70)";
  const visCh = result.channel_names;
  const testingTemporal = result.temporal_test ?? result.temporal;
  const temporalData = { ...testingTemporal, channel_names: result.channel_names };
  const totalDuration = Math.max(result.temporal.duration_s, 1e-6);
  const viewDuration = Math.max(totalDuration / Math.max(zoomX, 1), totalDuration / 5000);
  const panRange = Math.max(0, totalDuration - viewDuration);
  const viewStart = panNorm * panRange;

  return (
    <div className="space-y-5">
      {/* Classification metrics */}
      <div>
        <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-slate-500">
          Classification Performance
          {result.metrics.evaluation?.folds ? (
            <span className="ml-2 font-normal tracking-normal normal-case text-slate-600">
              ({result.metrics.evaluation.folds}-fold CV)
            </span>
          ) : null}
        </p>
        <div className="grid gap-3 sm:grid-cols-2">
          <MetricsCard m={result.metrics.csp_lda}   label="CSP + LDA"   accentColor={accentColor} />
          <MetricsCard m={result.metrics.fbcsp_lda} label="FBCSP + LDA" accentColor="rgba(251,146,60,0.70)" />
        </div>
      </div>

      {/* ROC */}
      <div>
        <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-500">ROC Curves</p>
        <RocChart result={result} />
      </div>

      {/* Temporal */}
      <div>
        <div className="mb-3 flex items-center justify-between">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">
              EEG · Temporal Signal
              <span className="ml-2 font-normal tracking-normal normal-case text-slate-600">(testing · full recording)</span>
            </p>
          </div>
          <div className="flex items-center gap-1">
            {(["raw", "filtered", "both"] as const).map(opt => (
              <button key={opt} onClick={() => setViewMode(opt)}
                className={cn(
                  "rounded-sm border px-2 py-1 text-[9px] font-semibold uppercase tracking-widest transition-all",
                  viewMode === opt
                    ? "border-indigo-500/30 bg-indigo-500/10 text-indigo-200"
                    : "border-transparent text-slate-600 hover:text-slate-400",
                )}>
                {opt}
              </button>
            ))}
          </div>
        </div>

        {/* Event legend */}
        <div className="mb-3 flex items-center gap-4 text-[9px] font-mono text-slate-600">
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block h-px w-4 bg-emerald-400/70" />left (+1)
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block h-px w-4 bg-red-400/70" />right (−1)
          </span>
        </div>

        {/* Controls */}
        <div className="mb-3 grid gap-2 sm:grid-cols-[160px_1fr]">
          <div className="rounded border border-white/[0.06] bg-white/[0.015] px-2.5 py-2">
            <p className="text-[9px] font-semibold uppercase tracking-widest text-slate-500">Zoom X</p>
            <div className="mt-1.5 flex items-center gap-2">
              <input type="range" min={1} max={20} step={0.5} value={zoomX}
                onChange={e => { setZoomX(Number(e.target.value)); setPanNorm(0); }}
                className="w-full" />
              <span className="w-8 shrink-0 text-right font-mono text-[10px] text-slate-400">{zoomX.toFixed(1)}×</span>
            </div>
          </div>
          <div className="rounded border border-white/[0.06] bg-white/[0.015] px-2.5 py-2">
            <p className="text-[9px] font-semibold uppercase tracking-widest text-slate-500">Timeline</p>
            <div className="mt-1.5 flex items-center gap-2">
              <input type="range" min={0} max={1000} step={1}
                value={Math.round(panNorm * 1000)}
                onChange={e => setPanNorm(Number(e.target.value) / 1000)}
                className="w-full" disabled={panRange <= 0} />
              <span className="w-20 shrink-0 text-right font-mono text-[10px] text-slate-400">
                {viewStart.toFixed(1)}–{(viewStart + viewDuration).toFixed(1)}s
              </span>
            </div>
          </div>
        </div>

        <TemporalChart
          windowSeconds={epochWindow}
          apiData={temporalData}
          visibleChannels={visCh}
          viewMode={viewMode}
          viewportStartSec={viewStart}
          viewportDurationSec={viewDuration}
        />
      </div>

      {/* Info strip */}
      <div className="flex items-center gap-3 border-t border-white/[0.04] pt-3 font-mono text-[9px] text-slate-600">
        <span>{result.n_epochs.train} train trials</span>
        <span>·</span>
        <span>{result.n_epochs.test} test trials</span>
        <span>·</span>
        <span>{result.fs} Hz</span>
        <span>·</span>
        <span>{result.temporal.duration_s}s epoch</span>
      </div>
    </div>
  );
}

// ─── Step Rail ────────────────────────────────────────────────────────────────
function StepRail({ currentStep }: { currentStep: number }) {
  const steps = [
    { n: 1, label: "Data Import"   },
    { n: 2, label: "Configuration" },
    { n: 3, label: "Analysis"      },
  ];
  return (
    <div className="flex items-center gap-0">
      {steps.map(({ n, label }, idx) => {
        const done   = n < currentStep;
        const active = n === currentStep;
        return (
          <div key={n} className="flex items-center">
            <div className="flex flex-col items-start gap-1">
              <div className="flex items-center gap-1.5">
                <span className={cn(
                  "inline-flex h-[3px] w-3 rounded-full transition-all duration-500",
                  done ? "bg-emerald-500/50" : active ? "bg-indigo-400/70" : "bg-white/[0.08]",
                )} />
                <span className={cn(
                  "text-[9px] font-semibold uppercase tracking-[0.14em] transition-colors duration-500",
                  done ? "text-emerald-400/60" : active ? "text-indigo-300/90" : "text-slate-600",
                )}>
                  {done ? <Check className="inline h-2.5 w-2.5 -mt-px" /> : null}
                  {!done ? label : label}
                </span>
              </div>
            </div>
            {idx < 2 && (
              <div className={cn(
                "mx-4 h-px w-12 transition-all duration-700",
                n < currentStep ? "bg-emerald-500/20" : "bg-white/[0.05]",
              )} />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function MvpSequence() {
  const [files, setFiles] = useState<Partial<Record<FileRole, UploadedFile>>>({});
  const [preResult,  setPreResult]  = useState<ApiResult | null>(null);
  const [postResult, setPostResult] = useState<ApiResult | null>(null);
  const [processing, setProcessing] = useState<"idle" | "pre" | "post">("idle");
  const [apiError,   setApiError]   = useState<string | null>(null);
  const [processKey, setProcessKey] = useState(0);

  const [filters, setFilters] = useState<Filters>({ highPassHz: 1, lowPassHz: 40 });
  const [hyperParams, setHyperParams] = useState<HyperParams>({
    epochTmin: 2.0,
    epochTmax: 6.0,
    cspComponents: 4,
    fbcspKBest: 12,
  });
  const [activeConfigTab, setActiveConfigTab] = useState<"filters" | "hyperparams">("filters");
  const [isConfigDirty, setIsConfigDirty] = useState(false);
  const [detailTab,  setDetailTab]  = useState<DetailTab>("pre");

  const allFilesLoaded = ["pre-train","pre-test","post-train","post-test"].every(r => files[r as FileRole]);
  const bothProcessed  = Boolean(preResult && postResult);
  const currentStep    = !allFilesLoaded ? 1 : !bothProcessed ? 2 : 3;

  function onFile(role: FileRole, file: File) {
    setFiles(prev => ({ ...prev, [role]: { role, file } }));
    setPreResult(null); setPostResult(null); setApiError(null); setIsConfigDirty(false);
  }

  function markDirty() {
    setPreResult(null); setPostResult(null); setApiError(null); setIsConfigDirty(true);
  }

  function setFilter<K extends keyof Filters>(key: K, value: Filters[K]) {
    setFilters(prev => prev[key] === value ? prev : { ...prev, [key]: value });
    markDirty();
  }

  function setHyper<K extends keyof HyperParams>(key: K, value: HyperParams[K]) {
    setHyperParams(prev => prev[key] === value ? prev : { ...prev, [key]: value });
    markDirty();
  }

  async function buildForm(trainFile: File, testFile: File): Promise<FormData> {
    const fd = new FormData();
    fd.append("training_file", trainFile);
    fd.append("test_file",     testFile);
    fd.append("high_pass_hz",   String(filters.highPassHz));
    fd.append("low_pass_hz",    String(filters.lowPassHz));
    fd.append("epoch_tmin",     String(hyperParams.epochTmin));
    fd.append("epoch_tmax",     String(hyperParams.epochTmax));
    fd.append("csp_components", String(hyperParams.cspComponents));
    fd.append("fbcsp_k_best",   String(hyperParams.fbcspKBest));
    fd.append("seed",           "42");
    fd.append("selected_channels", JSON.stringify(FIXED_CHANNELS));
    return fd;
  }

  async function handleProcess() {
    if (!allFilesLoaded) return;
    setApiError(null);

    try {
      setProcessing("pre");
      const preForm = await buildForm(files["pre-train"]!.file, files["pre-test"]!.file);
      const preRes  = await fetch(`${API}/process`, { method: "POST", body: preForm });
      if (!preRes.ok) throw new Error(`PRE: ${(await preRes.text()).slice(0, 200)}`);
      const preData: ApiResult = await preRes.json();

      setProcessing("post");
      const postForm = await buildForm(files["post-train"]!.file, files["post-test"]!.file);
      const postRes  = await fetch(`${API}/process`, { method: "POST", body: postForm });
      if (!postRes.ok) throw new Error(`POST: ${(await postRes.text()).slice(0, 200)}`);
      const postData: ApiResult = await postRes.json();

      setPreResult(preData);
      setPostResult(postData);
      setProcessKey(k => k + 1);
      setIsConfigDirty(false);
      setDetailTab("pre");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setApiError(msg.includes("fetch") ? "Backend unavailable — start the FastAPI server" : msg);
      setPreResult(null); setPostResult(null);
    } finally {
      setProcessing("idle");
    }
  }

  const epochWindow = Math.max(0.5, hyperParams.epochTmax - hyperParams.epochTmin);

  return (
    <div className="min-h-screen bg-[#05050c] text-slate-100">
      {/* Subtle vignette glow only — no grid */}
      <div className="pointer-events-none fixed inset-0" style={{
        background: "radial-gradient(ellipse 90% 50% at 50% -10%, rgba(99,102,241,0.05), transparent 70%)",
      }} />

      <div className="relative mx-auto max-w-5xl px-4 py-10 sm:px-6">

        {/* ── Header ──────────────────────────────────────────── */}
        <motion.header initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }} className="mb-10">

          {/* Metadata strip */}
          <div className="mb-4 flex items-center gap-3 font-mono">
            <span className="text-[9px] uppercase tracking-[0.22em] text-indigo-400/60">BR41N.IO</span>
            <span className="text-[9px] text-slate-500">·</span>
            <span className="text-[9px] uppercase tracking-[0.18em] text-slate-700">Hackathon 2026</span>
            <span className="flex-1 border-b border-white/[0.04]" />
            <span className="text-[9px] uppercase tracking-[0.18em] text-slate-700">Motor Imagery BCI</span>
          </div>

          <h1 className="text-[2.5rem] font-semibold leading-[1.1] tracking-[-0.02em] text-slate-50 sm:text-[3rem]">
            Stroke<br className="sm:hidden" />{" "}
            <span className="font-light text-slate-400">Rehabilitation</span>{" "}
            <span className="text-indigo-400">Analysis</span>
          </h1>
          <div className="mt-6 border-b border-white/[0.05] pb-6">
            <StepRail currentStep={currentStep} />
          </div>
        </motion.header>

        {/* ── Step 1: Upload ───────────────────────────────────── */}
        <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25, duration: 0.45 }}
          className="mb-3">

          <div className="rounded-md border border-white/[0.06] bg-white/[0.014]">
            {/* Panel header */}
            <div className="flex items-center justify-between border-b border-white/[0.05] px-5 py-3">
              <div className="flex items-center gap-2.5">
                <span className={cn(
                  "inline-block h-[3px] w-4 rounded-full",
                  allFilesLoaded ? "bg-emerald-500/60" : "bg-indigo-400/60",
                )} />
                <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-400">
                  Data Import
                </span>
              </div>
              <span className="font-mono text-[9px] text-slate-700">
                {Object.keys(files).length}/4 files
              </span>
            </div>

            <div className="p-5">
              <div className="grid gap-5 lg:grid-cols-2">
                {/* PRE column */}
                <div className="space-y-2">
                  <div className="mb-3 flex items-center gap-2">
                    <span className="inline-block h-[2px] w-3 rounded-full bg-indigo-400/50" />
                    <SectionLabel>Pre-Rehabilitation</SectionLabel>
                  </div>
                  <DropZone role="pre-train" label="Training" file={files["pre-train"] ?? null} onFile={onFile} />
                  <DropZone role="pre-test"  label="Test"     file={files["pre-test"]  ?? null} onFile={onFile} />
                </div>

                {/* Vertical divider */}
                <div className="hidden lg:block absolute left-1/2 h-full w-px bg-white/[0.04]" />

                {/* POST column */}
                <div className="space-y-2">
                  <div className="mb-3 flex items-center gap-2">
                    <span className="inline-block h-[2px] w-3 rounded-full bg-emerald-400/50" />
                    <SectionLabel accent="emerald">Post-Rehabilitation</SectionLabel>
                  </div>
                  <DropZone role="post-train" label="Training" file={files["post-train"] ?? null} onFile={onFile} />
                  <DropZone role="post-test"  label="Test"     file={files["post-test"]  ?? null} onFile={onFile} />
                </div>
              </div>
            </div>
          </div>
        </motion.section>

        {/* ── Step 2: Configure & Run ──────────────────────────── */}
        <AnimatePresence>
          {allFilesLoaded && (
            <motion.section key="s2"
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 6 }}
              transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              className="mb-3">

              <div className="rounded-md border border-white/[0.06] bg-white/[0.014]">
                {/* Panel header */}
                <div className="flex items-center justify-between border-b border-white/[0.05] px-5 py-3">
                  <div className="flex items-center gap-2.5">
                    <span className={cn(
                      "inline-block h-[3px] w-4 rounded-full",
                      bothProcessed ? "bg-emerald-500/60" : "bg-indigo-400/60",
                    )} />
                    <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-400">
                      Signal Processing
                    </span>
                    {isConfigDirty && (
                      <span className="rounded-sm border border-amber-500/20 bg-amber-500/10 px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider text-amber-400">
                        Stale — re-run
                      </span>
                    )}
                  </div>

                  <button onClick={handleProcess} disabled={processing !== "idle"}
                    className={cn(
                      "inline-flex items-center gap-2 rounded border px-3 py-1.5 text-[10px] font-semibold uppercase tracking-widest transition-all duration-250",
                      processing !== "idle"
                        ? "cursor-wait border-indigo-500/10 bg-indigo-500/[0.04] text-indigo-400/40"
                        : "border-indigo-500/25 bg-indigo-500/10 text-indigo-200 hover:border-indigo-400/40 hover:bg-indigo-500/18 hover:text-indigo-100",
                    )}>
                    {processing === "pre"  && <><Loader2 className="h-3 w-3 animate-spin" />Processing PRE…</>}
                    {processing === "post" && <><Loader2 className="h-3 w-3 animate-spin" />Processing POST…</>}
                    {processing === "idle" && <><Zap className="h-3 w-3" />Run Analysis</>}
                  </button>
                </div>

                <div className="p-5">
                  <div className="grid gap-5 lg:grid-cols-[240px_1fr]">
                    {/* Left: tabs + sliders */}
                    <div className="space-y-4">
                      {/* Tab switcher */}
                      <div className="flex rounded-sm border border-white/[0.06] bg-white/[0.01] p-[3px] gap-[3px]">
                        {(["filters", "hyperparams"] as const).map(tab => (
                          <button key={tab} onClick={() => setActiveConfigTab(tab)}
                            className={cn(
                              "flex-1 rounded-sm py-1.5 text-[9px] font-bold uppercase tracking-[0.14em] transition-all",
                              activeConfigTab === tab
                                ? "border border-indigo-400/25 bg-indigo-500/12 text-indigo-200"
                                : "text-slate-600 hover:text-slate-400",
                            )}>
                            {tab === "filters" ? "Filters" : "Hyperparams"}
                          </button>
                        ))}
                      </div>

                      {activeConfigTab === "filters" ? (
                        <div className="space-y-4">
                          <div>
                            <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-slate-500">Notch Filter</p>
                            <div className="rounded-sm border border-white/[0.06] bg-white/[0.01] px-3 py-2 font-mono text-[11px] text-slate-500">
                              50 + 60 Hz
                              <span className="ml-2 text-slate-600">fixed</span>
                            </div>
                          </div>
                          <Slider label="High-pass" value={filters.highPassHz} min={0}  max={20} unit="Hz" step={0.5} onChange={v => setFilter("highPassHz", v)} />
                          <Slider label="Low-pass"  value={filters.lowPassHz}  min={10} max={80} unit="Hz" step={0.5} onChange={v => setFilter("lowPassHz",  v)} />
                        </div>
                      ) : (
                        <div className="space-y-4">
                          <Slider label="Epoch tmin"     value={hyperParams.epochTmin}     min={0} max={4}  unit="s" step={0.1} onChange={v => setHyper("epochTmin",     Number(v.toFixed(1)))} />
                          <Slider label="Epoch tmax"     value={hyperParams.epochTmax}     min={4} max={8}  unit="s" step={0.1} onChange={v => setHyper("epochTmax",     Number(v.toFixed(1)))} />
                          <Slider label="CSP components" value={hyperParams.cspComponents} min={2} max={8}  unit=""  step={1}   onChange={v => setHyper("cspComponents", Math.round(v))} />
                          <Slider label="FBCSP k-best"   value={hyperParams.fbcspKBest}    min={4} max={24} unit=""  step={1}   onChange={v => setHyper("fbcspKBest",    Math.round(v))} />
                        </div>
                      )}
                    </div>

                    {/* Right: channel grid */}
                    <div>
                      <div className="mb-3 flex items-center justify-between">
                        <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">Electrode Montage</p>
                        <span className="font-mono text-[9px] text-slate-600">{FIXED_CHANNELS.length} channels · read-only</span>
                      </div>
                      <div className="grid grid-cols-6 gap-1">
                        {FIXED_CHANNELS.map((ch, i) => (
                          <div key={ch} className="rounded-sm border border-white/[0.06] bg-white/[0.015] py-1.5 text-center font-mono text-[10px]"
                            style={{ color: channelColor(i) }}>
                            {ch}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* ── Error banner ─────────────────────────────────────── */}
        <AnimatePresence>
          {apiError && (
            <motion.div key="err"
              initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}
              className="mb-3 flex items-start gap-3 rounded border border-red-500/15 bg-red-500/[0.05] px-4 py-3 text-[12px] text-red-400">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{apiError}</span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Step 3: Results ──────────────────────────────────── */}
        <AnimatePresence mode="wait">
          {bothProcessed && (
            <motion.section key={`s3-${processKey}`}
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 6 }}
              transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
              className="space-y-3">

              {/* ── Summary + Comparison ───────────────────────── */}
              <div className="rounded-md border border-white/[0.06] bg-white/[0.014]">
                <div className="flex items-center justify-between border-b border-white/[0.05] px-5 py-3">
                  <div className="flex items-center gap-2.5">
                    <span className="inline-block h-[3px] w-4 rounded-full bg-indigo-400/60" />
                    <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-400">
                      Rehabilitation Outcome
                    </span>
                  </div>
                </div>

                {/* Summary stats */}
                <div className="border-b border-white/[0.04] px-5 py-4">
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {[
                      { label: "Pre trials",   value: `${preResult!.n_epochs.train + preResult!.n_epochs.test}` },
                      { label: "Post trials",  value: `${postResult!.n_epochs.train + postResult!.n_epochs.test}` },
                      { label: "Sample rate",  value: `${preResult!.fs} Hz` },
                      { label: "Epoch",        value: `${preResult!.temporal.duration_s}s` },
                    ].map(({ label, value }) => (
                      <div key={label} className="rounded border border-white/[0.05] bg-[#0a0c14] px-3 py-2.5">
                        <p className="text-[8px] font-bold uppercase tracking-[0.18em] text-slate-700">{label}</p>
                        <p className="mt-0.5 font-mono text-[13px] tabular-nums text-slate-200">{value}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="p-5">
                  <ComparisonPanel pre={preResult!} post={postResult!} />
                </div>
              </div>

              {/* ── Detailed analysis (tabbed) ─────────────────── */}
              <div className="rounded-md border border-white/[0.06] bg-white/[0.014]">
                <div className="flex items-center justify-between border-b border-white/[0.05] px-5 py-3">
                  <div className="flex items-center gap-2.5">
                    <Activity className="h-3.5 w-3.5 text-slate-600" />
                    <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-400">
                      Session Detail
                    </span>
                  </div>
                  {/* Tab switcher */}
                  <div className="flex rounded-sm border border-white/[0.06] bg-white/[0.01] p-[3px] gap-[3px]">
                    {(["pre", "post"] as const).map(tab => (
                      <button key={tab} onClick={() => setDetailTab(tab)}
                        className={cn(
                          "rounded-sm px-3 py-1 text-[9px] font-bold uppercase tracking-[0.14em] transition-all",
                          detailTab === tab
                            ? tab === "pre"
                              ? "border border-indigo-400/25 bg-indigo-500/12 text-indigo-200"
                              : "border border-emerald-400/25 bg-emerald-500/12 text-emerald-200"
                            : "text-slate-600 hover:text-slate-400",
                        )}>
                        {tab === "pre" ? "Pre-Rehab" : "Post-Rehab"}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="p-5">
                  <AnimatePresence mode="wait">
                    <motion.div key={detailTab}
                      initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
                      transition={{ duration: 0.25 }}>
                      <SessionDetail
                        result={detailTab === "pre" ? preResult! : postResult!}
                        phase={detailTab}
                        epochWindow={epochWindow}
                      />
                    </motion.div>
                  </AnimatePresence>
                </div>
              </div>

            </motion.section>
          )}
        </AnimatePresence>

        {/* ── Footer ───────────────────────────────────────────── */}
        <motion.footer initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          transition={{ delay: 0.8, duration: 0.5 }}
          className="mt-10 flex items-center justify-between border-t border-white/[0.04] pt-5 font-mono text-[8px] uppercase tracking-[0.2em] text-slate-800">
          <span>BR41N.IO · Stroke Rehabilitation BCI · 2026</span>
          <span>.mat · .csv · .edf</span>
        </motion.footer>

      </div>
    </div>
  );
}
