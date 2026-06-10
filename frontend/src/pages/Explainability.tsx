import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api, type Analysis, type Patient } from '../api/client';
import { FeatureChart } from '../components/FeatureChart';
import { ArrowLeft, ShieldAlert, Cpu, Heart, Check, HelpCircle, Activity } from 'lucide-react';

export const Explainability: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [patient, setPatient] = useState<Patient | null>(null);
  const [radarData, setRadarData] = useState<any[]>([]);
  const [shapData, setShapData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedGroup, setSelectedGroup] = useState<string>('pitch_instability');

  useEffect(() => {
    if (!id) return;
    const fetchExplainData = async () => {
      try {
        setLoading(true);
        const anal = await api.getAnalysis(id);
        setAnalysis(anal);

        if (anal.patient_id) {
          const p = await api.getPatient(anal.patient_id);
          setPatient(p);
        }

        // Fetch explain endpoint for radar data and formatted SHAP values
        const explain = await api.getExplanation(id);
        setRadarData(explain.radar_data || []);

        // Process SHAP values for the bar chart
        if (anal.risk_breakdown) {
          const processedShaps: any[] = [];
          Object.entries(anal.risk_breakdown).forEach(([groupKey, group]: any) => {
            if (group.features) {
              group.features.forEach((f: any) => {
                processedShaps.push({
                  ...f,
                  group: groupKey,
                });
              });
            }
          });
          setShapData(processedShaps);
        }

        setLoading(false);
      } catch (err: any) {
        console.error("Error loading explanation data", err);
        setError("Failed to fetch SHAP explanation data. Verify the analysis exists and is complete.");
        setLoading(false);
      }
    };
    fetchExplainData();
  }, [id]);

  if (loading) {
    return (
      <div style={styles.loadingContainer}>
        <div className="spinner" />
        <p style={{ marginTop: '16px', color: 'var(--text-secondary)' }}>Calculating game-theoretic SHAP weights...</p>
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="page-layout fade-in" style={styles.errorContainer}>
        <div className="glass-card" style={styles.errorCard}>
          <ShieldAlert size={40} style={{ color: 'var(--red-400)' }} />
          <h2 style={{ fontSize: '1.2rem', marginTop: '12px' }}>Explainability Fetch Error</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', margin: '8px 0 16px' }}>{error || 'Data could not be loaded.'}</p>
          <Link to="/" className="btn btn-primary">Return Home</Link>
        </div>
      </div>
    );
  }

  const pName = patient?.name || 'Anonymized Evaluation';
  const pdProb = analysis.pd_probability !== undefined ? Math.round(analysis.pd_probability * 100) : 0;
  const riskGroups = analysis.risk_breakdown || {};
  const activeGroupData = riskGroups[selectedGroup];

  return (
    <div className="page-layout fade-in">
      {/* Header */}
      <div style={styles.header}>
        <Link to={`/report/${analysis.id}`} style={styles.backBtn}>
          <ArrowLeft size={16} />
          <span>Back to Report</span>
        </Link>
        <h1 style={styles.title}>Explainable AI (XAI) Assessment</h1>
        <p style={styles.subtitle}>
          Clinician explainability dashboard using SHAP values (Shapley Additive exPlanations) to trace machine learning logic.
        </p>
      </div>

      {/* Patient Summary Header bar */}
      <div className="glass-card" style={styles.summaryBar}>
        <div>
          <span style={styles.summaryLabel}>Patient</span>
          <span style={styles.summaryVal}>{pName}</span>
        </div>
        <div>
          <span style={styles.summaryLabel}>Predicted PD Likelihood</span>
          <span style={{ ...styles.summaryVal, color: pdProb >= 50 ? 'var(--red-400)' : 'var(--green-400)' }}>
            {pdProb}%
          </span>
        </div>
        <div>
          <span style={styles.summaryLabel}>Consensus Tier</span>
          <span style={styles.summaryVal} className="mono">{(analysis.risk_tier || 'N/A').toUpperCase().replace('_', ' ')}</span>
        </div>
        <div>
          <span style={styles.summaryLabel}>Evaluation ID</span>
          <span style={styles.summaryVal} className="mono" style={{ fontSize: '0.8rem' }}>{analysis.id.slice(0, 8)}...</span>
        </div>
      </div>

      {/* Biomarker Category Radar vs Highlight Description */}
      <div style={styles.grid2Col}>
        <div className="glass-card" style={styles.panel}>
          <h3 style={styles.panelTitle}>Biomarker Cluster Impairment Radar</h3>
          <p style={styles.panelDesc}>Aggregated risk mapping across clinical categories. Values represent scale percentages of abnormal SHAP contribution.</p>
          <FeatureChart type="radar" radarData={radarData} />
        </div>

        <div className="glass-card" style={styles.panel}>
          <h3 style={styles.panelTitle}>Clinical Risk Breakdown</h3>
          <p style={styles.panelDesc}>Select a category to view individual vocal component abnormalities.</p>

          <div style={styles.tabGrid}>
            {Object.entries(riskGroups).map(([key, group]: any) => {
              const isActive = key === selectedGroup;
              const riskPercent = Math.round(group.risk_score * 100);
              return (
                <button
                  key={key}
                  onClick={() => setSelectedGroup(key)}
                  style={{
                    ...styles.tabBtn,
                    borderColor: isActive ? 'var(--teal-500)' : 'var(--glass-border)',
                    background: isActive ? 'rgba(107, 79, 58, 0.08)' : 'rgba(255,255,255,0.01)',
                  }}
                >
                  <span style={styles.tabName}>{group.display}</span>
                  <span style={{ ...styles.tabVal, color: group.risk_score >= 0.5 ? 'var(--red-400)' : 'var(--green-400)' }}>
                    {riskPercent}% Risk
                  </span>
                </button>
              );
            })}
          </div>

          {activeGroupData && (
            <div style={styles.activeGroupDetails}>
              <h4 style={{ color: 'var(--teal-400)', fontSize: '0.95rem', fontWeight: 700 }}>
                {activeGroupData.display} Annotation
              </h4>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: 1.4 }}>
                {activeGroupData.description}
              </p>
              <div style={styles.clinicalNote}>
                <strong style={{ fontSize: '0.78rem', color: 'var(--text-primary)', display: 'block', marginBottom: '2px' }}>Clinician Note:</strong>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{activeGroupData.clinical_note}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* SHAP Bar Chart */}
      <div className="glass-card" style={{ ...styles.panel, marginTop: '24px' }}>
        <h3 style={styles.panelTitle}>Top Acoustic Feature Impact (SHAP Values)</h3>
        <p style={styles.panelDesc}>
          Horizontal bars indicate direction and strength of influence. <span style={{ color: 'var(--espresso)', fontWeight: 600 }}>Dark brown bars</span> push the classifier toward elevated screening risk. <span style={{ color: 'var(--walnut)', fontWeight: 600 }}>Walnut bars</span> push the classifier toward lower screening risk.
        </p>
        <FeatureChart type="shap" shapData={shapData} />
      </div>

      {/* Full Feature Grid table */}
      <div className="glass-card" style={{ ...styles.panel, marginTop: '24px' }}>
        <h3 style={styles.panelTitle}>Tabular Acoustic Biomarker Breakdown</h3>
        <p style={styles.panelDesc}>Comprehensive review of all 20+ extracted acoustic features, matched to clinical reference ranges.</p>

        <div style={{ overflowX: 'auto', marginTop: '16px' }}>
          <table style={styles.table}>
            <thead>
              <tr style={styles.thRow}>
                <th style={styles.th}>Category</th>
                <th style={styles.th}>Feature Name</th>
                <th style={styles.th}>Extracted Value</th>
                <th style={styles.th}>Impact (SHAP)</th>
                <th style={styles.th}>Reference Range Status</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(riskGroups).map(([groupKey, group]: any) => (
                <React.Fragment key={groupKey}>
                  {/* Category Header Row */}
                  <tr style={styles.groupHeaderRow}>
                    <td colSpan={5} style={styles.groupHeaderTd}>
                      {group.display}
                    </td>
                  </tr>
                  {group.features.map((feat: any) => {
                    const shapVal = feat.shap_contribution;
                    const isRisk = shapVal > 0;

                    let statusColor = 'var(--text-muted)';
                    let statusBg = 'rgba(255,255,255,0.02)';
                    if (feat.status === 'normal') { statusColor = 'var(--green-400)'; statusBg = 'rgba(232,221,212,0.38)'; }
                    else if (feat.status === 'borderline') { statusColor = 'var(--amber-400)'; statusBg = 'rgba(184,163,143,0.18)'; }
                    else if (feat.status === 'abnormal') { statusColor = 'var(--red-400)'; statusBg = 'rgba(58,42,32,0.10)'; }

                    return (
                      <tr key={feat.feature} style={styles.tr}>
                        <td style={styles.td} style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{group.display}</td>
                        <td style={styles.td} style={{ fontWeight: 600 }}>{feat.display_name}</td>
                        <td style={styles.td} className="mono">{feat.value !== null ? feat.value : 'N/A'}</td>
                        <td style={styles.td} className="mono">
                          <span style={{ color: isRisk ? 'var(--red-400)' : 'var(--green-400)', fontWeight: 700 }}>
                            {isRisk ? '+' : ''}{shapVal.toFixed(4)}
                          </span>
                        </td>
                        <td style={styles.td}>
                          <span style={{
                            ...styles.statusBadge,
                            color: statusColor,
                            background: statusBg,
                            borderColor: statusColor.replace(')', ', 0.3)'),
                          }}>
                            {feat.status}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  loadingContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 'calc(100vh - var(--nav-height))',
  },
  errorContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: 'calc(100vh - var(--nav-height))',
  },
  errorCard: {
    maxWidth: '400px',
    padding: '32px',
    textAlign: 'center',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  header: {
    marginBottom: '24px',
  },
  backBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '0.85rem',
    color: 'var(--teal-500)',
    textDecoration: 'none',
    marginBottom: '16px',
  },
  title: {
    fontSize: '1.8rem',
    fontWeight: 900,
    background: 'var(--gradient-teal)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    marginBottom: '6px',
  },
  subtitle: {
    fontSize: '0.92rem',
    color: 'var(--text-secondary)',
  },
  summaryBar: {
    padding: '16px 24px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '24px',
    background: 'rgba(11, 26, 46, 0.5)',
  },
  summaryLabel: {
    display: 'block',
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  summaryVal: {
    fontSize: '0.92rem',
    fontWeight: 700,
    color: 'var(--text-primary)',
    marginTop: '2px',
    display: 'block',
  },
  grid2Col: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '24px',
  },
  panel: {
    padding: '24px',
  },
  panelTitle: {
    fontSize: '1.1rem',
    fontWeight: 700,
    color: 'var(--text-primary)',
    marginBottom: '4px',
  },
  panelDesc: {
    fontSize: '0.8rem',
    color: 'var(--text-muted)',
    marginBottom: '20px',
    lineHeight: 1.4,
  },
  tabGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '12px',
    marginBottom: '20px',
  },
  tabBtn: {
    padding: '12px',
    border: '1px solid var(--glass-border)',
    borderRadius: 'var(--radius-md)',
    cursor: 'pointer',
    textAlign: 'left',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    transition: 'all var(--transition-fast)',
  },
  tabName: {
    fontSize: '0.82rem',
    fontWeight: 700,
    color: 'var(--text-primary)',
  },
  tabVal: {
    fontSize: '0.75rem',
    fontWeight: 600,
  },
  activeGroupDetails: {
    background: 'rgba(255,255,255,0.01)',
    border: '1px solid rgba(255,255,255,0.04)',
    borderRadius: 'var(--radius-md)',
    padding: '16px',
  },
  clinicalNote: {
    borderTop: '1px solid rgba(255,255,255,0.06)',
    paddingTop: '12px',
    marginTop: '12px',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    textAlign: 'left',
  },
  thRow: {
    borderBottom: '1px solid var(--glass-border)',
  },
  th: {
    padding: '12px 16px',
    fontSize: '0.75rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    fontWeight: 600,
  },
  groupHeaderRow: {
    background: 'rgba(0, 212, 170, 0.04)',
  },
  groupHeaderTd: {
    padding: '10px 16px',
    fontSize: '0.85rem',
    fontWeight: 700,
    color: 'var(--teal-400)',
    borderBottom: '1px solid rgba(107, 79, 58, 0.14)',
  },
  tr: {
    borderBottom: '1px solid rgba(255,255,255,0.02)',
    transition: 'background-color 0.15s ease',
  },
  td: {
    padding: '12px 16px',
    fontSize: '0.85rem',
  },
  statusBadge: {
    display: 'inline-flex',
    padding: '2px 8px',
    borderRadius: 'var(--radius-full)',
    fontSize: '0.72rem',
    fontWeight: 700,
    textTransform: 'uppercase',
    border: '1px solid transparent',
  },
};
