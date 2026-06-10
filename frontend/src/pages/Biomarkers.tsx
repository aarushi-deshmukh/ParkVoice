import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  ResponsiveContainer, Tooltip,
} from 'recharts';
import {
  Activity, Mic, Waves, BarChart2, AlertTriangle,
  RefreshCcw, Info
} from 'lucide-react';

/* ─────────────────────────────────────────────────────────────────────────────
   Biomarker group definitions
   ───────────────────────────────────────────────────────────────────────────── */
const BIOMARKER_GROUPS = [
  {
    key: 'pitch',
    label: 'Fundamental Frequency (F0)',
    icon: <Activity size={18} style={{ color: 'var(--teal-400)' }} />,
    color: 'var(--teal-400)',
    description: 'Average, max, and min F0 reflect vocal fold vibration regularity.',
    metrics: ['mdvp_fo_hz', 'mdvp_fhi_hz', 'mdvp_flo_hz'],
    labels: ['Avg F0 (Hz)', 'Max F0 (Hz)', 'Min F0 (Hz)'],
  },
  {
    key: 'jitter',
    label: 'Jitter — Cycle Variation',
    icon: <Waves size={18} style={{ color: 'var(--amber-400)' }} />,
    color: 'var(--amber-400)',
    description: 'Frequency perturbation metrics. Elevated jitter correlates with reduced vocal stability in PD.',
    metrics: ['mdvp_jitter_pct', 'mdvp_jitter_abs', 'mdvp_rap', 'mdvp_ppq', 'jitter_ddp'],
    labels: ['Jitter %', 'Jitter Abs', 'RAP', 'PPQ', 'DDP'],
  },
  {
    key: 'shimmer',
    label: 'Shimmer — Amplitude Variation',
    icon: <BarChart2 size={18} style={{ color: 'var(--purple-400)' }} />,
    color: 'var(--purple-400)',
    description: 'Amplitude perturbation measures. Higher shimmer indicates irregular vocal loudness control.',
    metrics: ['mdvp_shimmer', 'mdvp_shimmer_db', 'shimmer_apq3', 'shimmer_apq5', 'mdvp_apq', 'shimmer_dda'],
    labels: ['Shimmer', 'Shimmer dB', 'APQ3', 'APQ5', 'APQ', 'DDA'],
  },
  {
    key: 'noise',
    label: 'Noise-to-Harmonics Ratios',
    icon: <Mic size={18} style={{ color: 'var(--red-400)' }} />,
    color: 'var(--red-400)',
    description: 'NHR and HNR reflect the ratio of noise to tonal signal. Low HNR = high noise = PD indicator.',
    metrics: ['nhr', 'hnr'],
    labels: ['NHR', 'HNR (dB)'],
  },
  {
    key: 'nonlinear',
    label: 'Non-linear Dynamics',
    icon: <Activity size={18} style={{ color: 'var(--green-400)' }} />,
    color: 'var(--green-400)',
    description: 'RPDE and DFA capture fractal complexity and self-similarity in the vocal signal.',
    metrics: ['rpde', 'dfa'],
    labels: ['RPDE', 'DFA'],
  },
  {
    key: 'spread',
    label: 'Frequency Spread',
    icon: <Waves size={18} style={{ color: 'var(--teal-300)' }} />,
    color: 'var(--teal-300)',
    description: 'D2 and spread1/2 indicate the geometric signal complexity and non-linear frequency spread.',
    metrics: ['spread1', 'spread2', 'd2', 'ppe'],
    labels: ['Spread1', 'Spread2', 'D2', 'PPE'],
  },
];

/* ─────────────────────────────────────────────────────────────────────────────
   Radar chart data — normalise each group's metrics to a 0-100 score
   ───────────────────────────────────────────────────────────────────────────── */
const HEALTHY_NORMS: Record<string, [number, number]> = {
  mdvp_fo_hz: [88, 260], mdvp_fhi_hz: [102, 592], mdvp_flo_hz: [65, 239],
  mdvp_jitter_pct: [0, 0.03], mdvp_jitter_abs: [0, 0.0003], mdvp_rap: [0, 0.016],
  mdvp_ppq: [0, 0.014], jitter_ddp: [0, 0.048],
  mdvp_shimmer: [0, 0.12], mdvp_shimmer_db: [0, 1.2], shimmer_apq3: [0, 0.06],
  shimmer_apq5: [0, 0.08], mdvp_apq: [0, 0.1], shimmer_dda: [0, 0.17],
  nhr: [0, 0.025], hnr: [8, 34],
  rpde: [0.25, 0.69], dfa: [0.57, 0.82],
  spread1: [-8, -2], spread2: [0.007, 0.51], d2: [1.4, 3.7], ppe: [0.04, 0.55],
};

function normalise(key: string, value: number): number {
  const range = HEALTHY_NORMS[key];
  if (!range) return 50;
  const [lo, hi] = range;
  const pct = ((value - lo) / (hi - lo)) * 100;
  return Math.max(0, Math.min(100, parseFloat(pct.toFixed(1))));
}

/* ─────────────────────────────────────────────────────────────────────────────
   Component
   ───────────────────────────────────────────────────────────────────────────── */
export const Biomarkers: React.FC = () => {
  const [latestFeatures, setLatestFeatures] = useState<Record<string, number> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchLatest = async () => {
      try {
        setLoading(true);
        const response = await api.listAnalyses({ limit: 10 });
        const completed = response.items?.find((a: any) => a.status === 'complete' && a.features);
        if (completed?.features) {
          setLatestFeatures(completed.features);
        } else {
          setError('No completed analysis found. Upload a voice recording to populate biomarker data.');
        }
      } catch {
        setError('Failed to load biomarker data.');
      } finally {
        setLoading(false);
      }
    };
    fetchLatest();
  }, []);

  /* Radar summary across groups */
  const radarData = BIOMARKER_GROUPS.map((g) => {
    if (!latestFeatures) return { group: g.label.split(' ')[0], score: 0 };
    const scores = g.metrics
      .filter((m) => latestFeatures[m] !== undefined)
      .map((m) => normalise(m, latestFeatures[m]));
    const avg = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
    return { group: g.label.split('—')[0].trim(), score: parseFloat(avg.toFixed(1)) };
  });

  if (loading) {
    return (
      <div style={styles.center}>
        <div className="spinner" />
        <p style={{ marginTop: '16px', color: 'var(--text-secondary)' }}>Loading biomarker data…</p>
      </div>
    );
  }

  return (
    <div className="page-layout fade-in">
      {/* Header */}
      <section className="overview-hero page-hero">
        <div>
          <span className="eyebrow">Biomarker Dashboard</span>
          <h1>Voice Biomarker Dashboard</h1>
          <p>
            Acoustic feature profile extracted from the most recent completed voice screening session.
          </p>
        </div>
      </section>

      {/* Disclaimer */}
      <div style={styles.disclaimerBanner}>
        <AlertTriangle size={14} style={{ color: 'var(--amber-400)', flexShrink: 0, marginTop: '1px' }} />
        <span style={styles.disclaimerText}>
          <strong>Research Tool:</strong> Biomarker values are for research and exploratory screening
          support only. They are not diagnostic outputs and must not substitute clinical assessment.
        </span>
      </div>

      {error ? (
        <div style={styles.errorBox}>
          <Info size={18} style={{ color: 'var(--text-muted)' }} />
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>{error}</p>
        </div>
      ) : (
        <div style={styles.mainGrid}>
          {/* Left: Radar */}
          <div style={styles.radarPanel}>
            <div className="glass-card" style={styles.radarCard}>
              <h3 style={styles.cardTitle}>
                <BarChart2 size={18} style={{ color: 'var(--teal-400)' }} />
                <span>Acoustic Group Profile</span>
              </h3>
              <p style={styles.cardDesc}>
                Normalised scores (0–100) for each biomarker group relative to population reference ranges.
              </p>
              <div style={{ width: '100%', height: 300, marginTop: '16px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.06)" />
                    <PolarAngleAxis dataKey="group" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
                    <Radar
                      name="Biomarker Score"
                      dataKey="score"
                      stroke="var(--teal-500)"
                      fill="var(--teal-500)"
                      fillOpacity={0.18}
                      strokeWidth={2}
                    />
                    <Tooltip
                      contentStyle={{
                        background: 'rgba(6,15,30,0.95)',
                        border: '1px solid var(--glass-border)',
                        borderRadius: 'var(--radius-md)',
                        fontSize: '0.8rem',
                        color: 'var(--text-primary)',
                      }}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Right: Feature Groups */}
          <div style={styles.groupsCol}>
            {BIOMARKER_GROUPS.map((group) => (
              <div key={group.key} className="glass-card" style={styles.groupCard}>
                <div style={styles.groupHeader}>
                  {group.icon}
                  <div>
                    <span style={{ fontWeight: 700, fontSize: '0.92rem' }}>{group.label}</span>
                    <p style={styles.groupDesc}>{group.description}</p>
                  </div>
                </div>
                <div style={styles.metricGrid}>
                  {group.metrics.map((metricKey, idx) => {
                    const raw = latestFeatures?.[metricKey];
                    const pct = raw !== undefined ? normalise(metricKey, raw) : null;
                    return (
                      <div key={metricKey} style={styles.metricItem}>
                        <div style={styles.metricHeader}>
                          <span style={styles.metricLabel}>{group.labels[idx]}</span>
                          <span style={{ ...styles.metricValue, color: group.color }}>
                            {raw !== undefined ? raw.toFixed(4) : 'N/A'}
                          </span>
                        </div>
                        <div className="progress-bar" style={{ height: '3px', marginTop: '4px' }}>
                          <div
                            className="progress-fill"
                            style={{
                              width: pct !== null ? `${pct}%` : '0%',
                              background: group.color,
                              transition: 'width 0.6s ease',
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  center: {
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', minHeight: 'calc(100vh - var(--nav-height))',
  },
  pageHeader: {
    marginBottom: '16px',
    display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
  },
  title: {
    fontSize: '1.9rem', fontWeight: 950,
    background: 'var(--gradient-teal)',
    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
    marginBottom: '4px',
  },
  subtitle: { fontSize: '0.88rem', color: 'var(--text-secondary)' },
  disclaimerBanner: {
    display: 'flex', alignItems: 'flex-start', gap: '10px',
    background: 'rgba(245, 158, 11, 0.06)',
    border: '1px solid rgba(245, 158, 11, 0.25)',
    borderRadius: 'var(--radius-md)',
    padding: '10px 16px', marginBottom: '24px',
  },
  disclaimerText: { fontSize: '0.78rem', color: 'var(--amber-300)', lineHeight: 1.45 },
  errorBox: {
    display: 'flex', gap: '12px', alignItems: 'flex-start',
    background: 'rgba(255,255,255,0.02)',
    border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: 'var(--radius-md)', padding: '24px',
  },
  mainGrid: {
    display: 'grid', gridTemplateColumns: '380px 1fr', gap: '24px', alignItems: 'start',
  },
  radarPanel: { display: 'flex', flexDirection: 'column', gap: '24px' },
  radarCard: { padding: '24px' },
  cardTitle: {
    fontSize: '1rem', fontWeight: 700,
    display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px',
  },
  cardDesc: { fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.4 },
  groupsCol: { display: 'flex', flexDirection: 'column', gap: '16px' },
  groupCard: { padding: '20px' },
  groupHeader: { display: 'flex', gap: '12px', alignItems: 'flex-start', marginBottom: '16px' },
  groupDesc: { fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: '2px', lineHeight: 1.4 },
  metricGrid: {
    display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: '12px',
  },
  metricItem: { display: 'flex', flexDirection: 'column' },
  metricHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  metricLabel: { fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.03em' },
  metricValue: { fontSize: '0.8rem', fontWeight: 700 },
};
