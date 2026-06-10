import React, { useEffect, useState } from 'react';
import { AlertTriangle, Cpu } from 'lucide-react';
import { api, type EdgeBenchmarkResult } from '../api/client';

export const Benchmarks: React.FC = () => {
  const [data, setData] = useState<EdgeBenchmarkResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getEdgeBenchmarks().then(setData).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page-layout"><div className="panel">Loading benchmark measurements...</div></div>;

  return (
    <div className="page-layout overview-page">
      <section className="overview-hero">
        <div>
          <span className="eyebrow">ONNX Edge Inference</span>
          <h1>Benchmark Reporting</h1>
          <p>Only real host or target-hardware measurements are displayed. Raspberry Pi and Jetson figures require runs on those devices.</p>
        </div>
      </section>

      <div className="disclaimer"><AlertTriangle size={16} /> {data?.message ?? 'Real benchmark measurements only.'}</div>

      <section className="panel table-panel">
        <h2>Measured ONNX INT8 Results</h2>
        {data?.profiles?.length ? (
          <table>
            <thead><tr><th>Model</th><th>Device</th><th>Latency</th><th>Model Size</th><th>Memory</th><th>Compression</th><th>Status</th></tr></thead>
            <tbody>
              {data.profiles.map((profile) => (
                <tr key={`${profile.model}-${profile.device}`}>
                  <td>{profile.model ?? 'model'}</td>
                  <td><Cpu size={14} /> {profile.device}</td>
                  <td className="mono">{profile.latency_ms} ms</td>
                  <td className="mono">{profile.model_size_mb} MB</td>
                  <td className="mono">{profile.memory_usage_mb ?? 'n/a'} MB</td>
                  <td className="mono">{profile.compression_ratio ?? 'n/a'}</td>
                  <td><span className={`status ${profile.deployable ? 'normal' : 'abnormal'}`}>{profile.deployable ? 'Deployable' : 'Review'}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p className="muted">No benchmark file is available yet. Run `python backend/evaluation/edge_benchmark.py` on the hardware you want to report.</p>}
      </section>
    </div>
  );
};
