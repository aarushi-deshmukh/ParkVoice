import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertTriangle, BarChart3, Brain, Gauge, Mic, ShieldCheck, Timer } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api, type Analysis, type EdgeBenchmarkResult } from '../api/client';

const DISCLAIMER = 'This system is intended for research and screening support purposes only and is not a diagnostic medical device.';

const demoAnalysis: Analysis = {
  id: 'demo',
  status: 'complete',
  filename: 'demo-sustained-vowel.wav',
  created_at: new Date().toISOString(),
  pd_probability: 0.37,
  risk_tier: 'moderate',
  confidence: 0.78,
  uncertainty_score: 0.22,
  severity_score: 18.4,
  predicted_updrs: 18.4,
  lower_bound: 12.1,
  upper_bound: 25.0,
  quality: { quality_score: 0.82, quality_category: 'Good', snr: 22.4, duration: 8.2, clipping: 0.001, silence_ratio: 0.12, signal_stability: 0.79, warnings: [] },
  biomarkers: [
    { key: 'jitter_local', name: 'Jitter', value: 0.84, healthy_range: '0.00-1.00%', status: 'Normal', explanation: 'Elevated jitter may indicate reduced vocal fold stability.' },
    { key: 'shimmer_local', name: 'Shimmer', value: 4.42, healthy_range: '0.00-3.80%', status: 'Borderline', explanation: 'Elevated shimmer may indicate irregular vocal loudness control.' },
    { key: 'hnr', name: 'HNR', value: 17.1, healthy_range: '18-35 dB', status: 'Borderline', explanation: 'Reduced HNR may indicate increased breathiness.' },
    { key: 'nhr', name: 'NHR', value: 0.018, healthy_range: '0.00-0.02', status: 'Normal', explanation: 'Elevated NHR may indicate a noisier phonation signal.' },
    { key: 'rpde', name: 'RPDE', value: 0.49, healthy_range: '0.20-0.45', status: 'Borderline', explanation: 'Elevated RPDE may indicate less regular vocal dynamics.' },
    { key: 'dfa', name: 'DFA', value: 0.71, healthy_range: '0.50-0.75', status: 'Normal', explanation: 'DFA outside the reference band may indicate altered signal complexity.' },
    { key: 'ppe', name: 'PPE', value: 0.19, healthy_range: '0.05-0.15', status: 'Borderline', explanation: 'Elevated PPE may indicate unstable pitch period control.' },
    { key: 'pitch_cv', name: 'Pitch Variability', value: 0.11, healthy_range: '0.00-0.12', status: 'Normal', explanation: 'Elevated pitch variability may reflect reduced phonatory steadiness.' },
    { key: 'd2', name: 'Signal Complexity', value: 2.3, healthy_range: '1.40-3.70', status: 'Normal', explanation: 'Reduced signal complexity may reflect a less flexible vocal pattern.' },
  ],
  clinical_explanations: [
    { label: 'HNR', value: 17.1, status: 'Borderline', clinical_explanation: 'Reduced HNR may indicate increased breathiness.', shap_value: 0.041 },
    { label: 'Shimmer', value: 4.42, status: 'Borderline', clinical_explanation: 'Elevated shimmer may indicate irregular vocal loudness control.', shap_value: 0.033 },
    { label: 'PPE', value: 0.19, status: 'Borderline', clinical_explanation: 'Elevated PPE may indicate unstable pitch period control.', shap_value: 0.025 },
  ],
};

export const Overview: React.FC = () => {
  const [analysis, setAnalysis] = useState<Analysis>(demoAnalysis);
  const [benchmarks, setBenchmarks] = useState<EdgeBenchmarkResult | null>(null);

  useEffect(() => {
    api.listAnalyses({ limit: 8 })
      .then((res) => {
        const latest = res.items.find((item) => item.status === 'complete');
        if (latest) setAnalysis({ ...demoAnalysis, ...latest });
      })
      .catch(() => undefined);
    api.getEdgeBenchmarks().then(setBenchmarks).catch(() => undefined);
  }, []);

  const risk = Math.round((analysis.pd_probability ?? 0) * 100);
  const confidence = Math.round((analysis.confidence ?? 0) * 100);
  const uncertainty = Math.round((analysis.uncertainty_score ?? analysis.uncertainty?.uncertainty_score ?? 0) * 100);
  const quality = analysis.quality ?? {};
  const biomarkers = analysis.biomarkers ?? demoAnalysis.biomarkers ?? [];
  const insights = analysis.clinical_explanations ?? demoAnalysis.clinical_explanations ?? [];

  const chartData = useMemo(() => biomarkers.map((b) => ({ name: b.name, value: Number(b.value), status: b.status })), [biomarkers]);

  return (
    <div className="page-layout overview-page">
      <section className="overview-hero">
        <div>
          <span className="eyebrow">AI-Assisted Parkinson's Screening</span>
          <h1>Parkinson's Risk Assessment</h1>
          <p>Acoustic screening support with calibrated ensemble risk assessment, conformal severity intervals, SHAP insights, and edge-readiness reporting.</p>
        </div>
        <Link to="/home" className="btn btn-primary"><Mic size={18} /> Start Screening</Link>
      </section>

      <div className="disclaimer"><AlertTriangle size={16} /> {DISCLAIMER}</div>

      <section className="metric-grid">
        <article className="metric-card"><Gauge size={20} /><span>Risk Score</span><strong>{risk}%</strong><small>{analysis.risk_tier?.replace('_', ' ') ?? 'pending'}</small></article>
        <article className="metric-card"><ShieldCheck size={20} /><span>Confidence</span><strong>{confidence}%</strong><small>calibrated agreement</small></article>
        <article className="metric-card"><Brain size={20} /><span>Uncertainty</span><strong>{uncertainty}%</strong><small>lower is better</small></article>
        <article className="metric-card"><Timer size={20} /><span>Audio Quality</span><strong>{quality.quality_category ?? 'Unknown'}</strong><small>{Math.round((quality.quality_score ?? 0) * 100)}% quality score</small></article>
      </section>

      <section className="dashboard-grid">
        <article className="panel risk-panel">
          <h2>Severity Estimation</h2>
          <div className="severity-value">{analysis.predicted_updrs ?? analysis.severity_score ?? 0}</div>
          <p>Conformal UPDRS interval: {analysis.lower_bound ?? analysis.severity_lower_ci ?? 0} to {analysis.upper_bound ?? analysis.severity_upper_ci ?? 0}</p>
          <div className="interval-track"><span style={{ left: '12%', width: '38%' }} /></div>
        </article>

        <article className="panel">
          <h2>Biomarker Profile</h2>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData.slice(0, 9)}>
                <CartesianGrid stroke="#E8DDD4" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#6B4F3A' }} interval={0} angle={-20} textAnchor="end" height={64} />
                <YAxis tick={{ fontSize: 11, fill: '#6B4F3A' }} />
                <Tooltip />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry) => <Cell key={entry.name} fill={entry.status === 'Abnormal' ? '#3A2A20' : entry.status === 'Borderline' ? '#6B4F3A' : '#8C735F'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel table-panel">
          <h2>Clinically Explainable Biomarkers</h2>
          <table>
            <thead><tr><th>Biomarker</th><th>Value</th><th>Healthy Range</th><th>Status</th></tr></thead>
            <tbody>
              {biomarkers.map((b) => (
                <tr key={b.key}><td>{b.name}</td><td className="mono">{b.value}</td><td>{b.healthy_range}</td><td><span className={`status ${b.status.toLowerCase()}`}>{b.status}</span></td></tr>
              ))}
            </tbody>
          </table>
        </article>

        <article className="panel">
          <h2>SHAP Insights</h2>
          <div className="insight-list">
            {insights.map((item) => (
              <div className="insight" key={item.label}>
                <strong>{item.label}</strong>
                <span>{item.clinical_explanation}</span>
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <h2>Benchmark Results</h2>
          {benchmarks?.profiles?.length ? benchmarks.profiles.slice(0, 4).map((profile) => (
            <div className="benchmark-row" key={`${profile.model}-${profile.device}`}>
              <BarChart3 size={16} />
              <span>{profile.model ?? profile.device}</span>
              <strong>{profile.latency_ms} ms</strong>
            </div>
          )) : <p className="muted">No real benchmark measurements found. Run the benchmark script on the target hardware.</p>}
        </article>
      </section>
    </div>
  );
};
