import React, { useEffect, useState } from 'react';
import { useStore } from '../store/useStore';
import { api, type Patient, type PatientHistory, type PatientTrends } from '../api/client';
import { Link } from 'react-router-dom';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  Users, UserPlus, FileText, ChevronRight,
  TrendingUp, TrendingDown, RefreshCcw,
  Sparkles, Calendar, Activity, AlertTriangle
} from 'lucide-react';

export const History: React.FC = () => {
  const { patients, selectedPatientId, setSelectedPatientId, fetchPatients } = useStore();

  const [history, setHistory] = useState<PatientHistory | null>(null);
  const [trends, setTrends] = useState<PatientTrends | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [showAddPatient, setShowAddPatient] = useState(false);
  const [newPatient, setNewPatient] = useState({ name: '', age: '', gender: 'Male', notes: '' });

  useEffect(() => {
    fetchPatients();
  }, [fetchPatients]);

  // Load history and trends when selectedPatientId changes
  useEffect(() => {
    if (!selectedPatientId) {
      setHistory(null);
      setTrends(null);
      return;
    }

    const fetchHistoryData = async () => {
      setLoadingHistory(true);
      try {
        const hist = await api.getPatientHistory(selectedPatientId);
        setHistory(hist);
        const trnd = await api.getPatientTrends(selectedPatientId);
        setTrends(trnd);
        setLoadingHistory(false);
      } catch (err) {
        console.error("Failed to load patient history", err);
        setLoadingHistory(false);
      }
    };
    fetchHistoryData();
  }, [selectedPatientId]);

  const handleAddPatientSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newPatient.name.trim()) return;
    try {
      const created = await api.createPatient({
        name: newPatient.name,
        age: newPatient.age ? parseInt(newPatient.age, 10) : undefined,
        gender: newPatient.gender,
        notes: newPatient.notes || undefined,
      });
      fetchPatients();
      setSelectedPatientId(created.id);
      setShowAddPatient(false);
      setNewPatient({ name: '', age: '', gender: 'Male', notes: '' });
    } catch (err) {
      console.error(err);
    }
  };

  const getTrendIcon = (trend: string) => {
    if (trend === 'increasing') return <TrendingUp size={16} style={{ color: 'var(--red-400)' }} />;
    if (trend === 'decreasing') return <TrendingDown size={16} style={{ color: 'var(--green-400)' }} />;
    return <TrendingDown size={16} style={{ color: 'var(--text-muted)', transform: 'rotate(-90deg)' }} />;
  };

  const activePatient = patients.find(p => p.id === selectedPatientId);

  // Format Recharts trend data
  const chartData = trends && trends.dates ? trends.dates.map((date, idx) => {
    const formattedDate = new Date(date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    return {
      date: formattedDate,
      probability: Math.round(trends.pd_probability[idx] * 100),
      severity: trends.severity_score[idx] ? parseFloat(trends.severity_score[idx].toFixed(1)) : 0,
    };
  }) : [];

  return (
    <div className="page-layout fade-in">
      <section className="overview-hero page-hero">
        <div>
          <span className="eyebrow">Longitudinal Screening</span>
          <h1>Patient Directory & Tracker</h1>
          <p>
          Longitudinal screening support for tracking voice-based PD risk assessments and UPDRS severity trends across evaluation sessions.
          </p>
        </div>
      </section>

      <div style={styles.mainGrid}>
        {/* Left Side: Patient list */}
        <div style={styles.sidebar}>
          <div className="glass-card" style={styles.sidebarCard}>
            <div style={styles.sidebarHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Users size={18} style={{ color: 'var(--teal-500)' }} />
                <span style={{ fontWeight: 700 }}>Profiles ({patients.length})</span>
              </div>
              <button onClick={() => setShowAddPatient(!showAddPatient)} className="btn btn-secondary btn-sm" style={{ padding: '4px 8px' }}>
                <UserPlus size={14} />
              </button>
            </div>

            {showAddPatient && (
              <form onSubmit={handleAddPatientSubmit} style={styles.addPatientForm}>
                <input
                  type="text"
                  required
                  placeholder="Patient Name"
                  value={newPatient.name}
                  onChange={(e) => setNewPatient({ ...newPatient, name: e.target.value })}
                  className="input"
                  style={{ fontSize: '0.8rem', padding: '8px 10px' }}
                />
                <div style={{ display: 'flex', gap: '8px' }}>
                  <input
                    type="number"
                    placeholder="Age"
                    value={newPatient.age}
                    onChange={(e) => setNewPatient({ ...newPatient, age: e.target.value })}
                    className="input"
                    style={{ fontSize: '0.8rem', padding: '8px 10px', flex: 1 }}
                  />
                  <select
                    value={newPatient.gender}
                    onChange={(e) => setNewPatient({ ...newPatient, gender: e.target.value })}
                    className="input"
                    style={{ fontSize: '0.8rem', padding: '8px 10px', flex: 1 }}
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
                <button type="submit" className="btn btn-primary btn-sm" style={{ width: '100%', justifyContent: 'center' }}>
                  Register Patient
                </button>
              </form>
            )}

            <div style={styles.patientList}>
              {patients.length === 0 ? (
                <div style={styles.emptySidebar}>No patients registered.</div>
              ) : (
                patients.map((p) => {
                  const isActive = p.id === selectedPatientId;
                  const prob = p.latest_pd_probability !== undefined && p.latest_pd_probability !== null 
                    ? `${Math.round(p.latest_pd_probability * 100)}%` 
                    : 'N/A';
                  return (
                    <div
                      key={p.id}
                      onClick={() => setSelectedPatientId(p.id)}
                      style={{
                        ...styles.patientItem,
                        background: isActive ? 'rgba(107, 79, 58, 0.08)' : 'transparent',
                        borderColor: isActive ? 'var(--teal-500)' : 'transparent',
                      }}
                    >
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', flex: 1 }}>
                        <span style={styles.patientName}>{p.name}</span>
                        <span style={styles.patientMeta}>Age {p.age || 'N/A'} | {p.gender}</span>
                      </div>
                      <div style={styles.patientStatusBadge}>
                        <span style={{ fontSize: '0.78rem', fontWeight: 700, color: p.latest_pd_probability && p.latest_pd_probability >= 0.5 ? 'var(--red-400)' : 'var(--green-400)' }}>
                          {prob}
                        </span>
                        <ChevronRight size={14} style={{ color: 'var(--text-muted)' }} />
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>

        {/* Right Side: Longitudinal detail and transformer forecast */}
        <div style={styles.contentArea}>
          {!selectedPatientId ? (
            <div className="glass-card" style={styles.emptyContent}>
              <Users size={40} style={{ color: 'var(--text-muted)' }} />
              <h3 style={{ fontSize: '1.1rem', marginTop: '12px' }}>Select a Patient Profile</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '4px' }}>
                Select a patient from the list on the left to review their longitudinal trend data and progression models.
              </p>
            </div>
          ) : loadingHistory ? (
            <div style={styles.loaderArea}>
              <div className="spinner" />
              <p style={{ marginTop: '16px', color: 'var(--text-secondary)' }}>Compiling longitudinal database...</p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              
              {/* Patient Header Block */}
              {activePatient && (
                <div className="glass-card" style={styles.patientBanner}>
                  <div>
                    <h2 style={{ fontSize: '1.3rem', fontWeight: 800 }}>{activePatient.name}</h2>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>ID: <span className="mono">{activePatient.id}</span></span>
                  </div>
                  <div style={styles.bannerMeta}>
                    <div style={styles.bannerMetaItem}>
                      <span style={styles.bmLabel}>Age</span>
                      <span style={styles.bmValue}>{activePatient.age || 'N/A'}</span>
                    </div>
                    <div style={styles.bannerMetaItem}>
                      <span style={styles.bmLabel}>Gender</span>
                      <span style={styles.bmValue}>{activePatient.gender || 'N/A'}</span>
                    </div>
                    <div style={styles.bannerMetaItem}>
                      <span style={styles.bmLabel}>Assessments</span>
                      <span style={styles.bmValue}>{history?.recordings.length || 0}</span>
                    </div>
                  </div>
                </div>
              )}

              {history && history.recordings.length === 0 ? (
                <div className="glass-card" style={styles.emptyContent}>
                  <Calendar size={36} style={{ color: 'var(--text-muted)' }} />
                  <h3 style={{ fontSize: '1rem', marginTop: '12px' }}>No Evaluation History</h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '4px', maxWidth: '400px' }}>
                    This patient profile does not have any completed evaluations. Navigate to the <strong>Upload</strong> or <strong>Record</strong> tab to perform an assessment.
                  </p>
                </div>
              ) : (
                <>
                  {/* Trends stats summary widgets */}
                  {history && history.trends && (
                    <div style={styles.trendsSummaryRow}>
                      <div className="glass-card" style={styles.trendSummaryCard}>
                        <span style={styles.trendSummaryLabel}>PD Probability Trend</span>
                        <div style={styles.trendRow}>
                          {getTrendIcon(history.trends.pd_probability)}
                          <span style={styles.trendVal} className="mono">
                            {history.trends.pd_probability.toUpperCase()}
                          </span>
                        </div>
                      </div>
                      <div className="glass-card" style={styles.trendSummaryCard}>
                        <span style={styles.trendSummaryLabel}>UPDRS Severity Trend</span>
                        <div style={styles.trendRow}>
                          {getTrendIcon(history.trends.severity)}
                          <span style={styles.trendVal} className="mono">
                            {history.trends.severity.toUpperCase()}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Recharts Longitudinal Line Charts */}
                  <div className="glass-card" style={styles.chartPanel}>
                    <h3 style={styles.panelTitle}>Longitudinal Screening Progression</h3>
                    <p style={styles.panelDesc}>Time-series mapping showing predicted PD risk assessment probability (%) and UPDRS severity score over recent evaluation sessions.</p>
                    
                    <div style={{ width: '100%', height: 300, marginTop: '16px' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                          <XAxis dataKey="date" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                          <YAxis yAxisId="left" domain={[0, 100]} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} label={{ value: 'PD Probability (%)', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', style: {textAnchor: 'middle'}, offset: 15 }} />
                          <YAxis yAxisId="right" orientation="right" domain={[0, 108]} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} label={{ value: 'UPDRS Score', angle: 90, position: 'insideRight', fill: 'var(--text-muted)', style: {textAnchor: 'middle'}, offset: 15 }} />
                          <Tooltip contentStyle={styles.tooltipStyle} />
                          <Legend verticalAlign="top" height={36} />
                          <Line yAxisId="left" type="monotone" dataKey="probability" stroke="var(--teal-500)" activeDot={{ r: 6 }} name="PD Probability (%)" strokeWidth={2.5} />
                          <Line yAxisId="right" type="monotone" dataKey="severity" stroke="var(--amber-500)" name="UPDRS Score" strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Experimental Progression Sandbox Notice */}
                  {history && (
                    <div className="glass-card" style={{ ...styles.chartPanel, overflow: 'hidden' }}>
                      <div style={styles.panelHeaderRow}>
                        <div>
                          <h3 style={{ ...styles.panelTitle, display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Sparkles size={18} style={{ color: 'var(--purple-400)' }} />
                            <span>Progression Forecasting — Experimental</span>
                          </h3>
                          <p style={styles.panelDesc}>
                            Transformer-based longitudinal progression forecasting is available as a research prototype in the experimental sandbox.
                          </p>
                        </div>
                      </div>

                      {history.progression_prediction && !history.progression_prediction.error ? (
                        <div style={styles.progressionDetailGrid}>
                          <div style={styles.forecastMetricCard}>
                            <span style={styles.forecastLabel}>Progression Risk</span>
                            <span style={styles.forecastVal} className="mono">
                              {Math.round(history.progression_prediction.progression_risk * 100)}%
                            </span>
                          </div>
                          <div style={styles.forecastMetricCard}>
                            <span style={styles.forecastLabel}>Predicted 6m Trend</span>
                            <span
                              style={{
                                ...styles.progressionBadge,
                                color:
                                  history.progression_prediction.trend === 'Improving'
                                    ? 'var(--green-400)'
                                    : history.progression_prediction.trend === 'Worsening'
                                    ? 'var(--red-400)'
                                    : 'var(--text-primary)',
                              }}
                            >
                              {history.progression_prediction.trend}
                            </span>
                          </div>
                          <div style={styles.forecastMetricCard}>
                            <span style={styles.forecastLabel}>Trend Confidence</span>
                            <span style={styles.forecastVal} className="mono">
                              {Math.round(history.progression_prediction.trend_confidence * 100)}%
                            </span>
                          </div>
                        </div>
                      ) : (
                        <div style={styles.experimentalNoticeBox}>
                          <AlertTriangle size={20} style={{ color: 'var(--purple-400)', flexShrink: 0 }} />
                          <div>
                            <h4 style={{ color: 'var(--purple-300)', fontWeight: 700, fontSize: '0.9rem' }}>Research Prototype — Not Clinically Validated</h4>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: '4px', lineHeight: 1.4 }}>
                              The transformer-based 6-month progression model is an <strong>experimental feature</strong> located in
                              <code style={{ background: 'rgba(107,79,58,0.12)', padding: '1px 6px', borderRadius: '4px', fontSize: '0.78rem' }}>
                                {' '}backend/research/experimental/{' '}
                              </code>.
                              It requires a minimum of <strong>3 separate evaluation sessions</strong> and produces results for research purposes only.
                              Outputs must not be interpreted as clinical prognosis.
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Past evaluations table */}
                  <div className="glass-card" style={styles.chartPanel}>
                    <h3 style={styles.panelTitle}>Historical Evaluations Log</h3>
                    <div style={{ overflowX: 'auto', marginTop: '12px' }}>
                      <table style={styles.table}>
                        <thead>
                          <tr style={styles.trHead}>
                            <th style={styles.th}>Date</th>
                            <th style={styles.th}>Filename</th>
                            <th style={styles.th}>PD Prob</th>
                            <th style={styles.th}>UPDRS III</th>
                            <th style={styles.th}>Risk Assessment</th>
                            <th style={styles.th} style={{ textAlign: 'right' }}>Actions</th>
                          </tr>
                        </thead>
                        <tbody>
                          {history.recordings.map((rec) => {
                            const date = new Date(rec.date).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' });
                            let badgeClass = 'low';
                            if (rec.risk_tier === 'moderate') badgeClass = 'moderate';
                            else if (rec.risk_tier === 'high') badgeClass = 'high';
                            else if (rec.risk_tier === 'very_high') badgeClass = 'very_high';

                            return (
                              <tr key={rec.id} style={styles.tr}>
                                <td style={styles.td} className="mono">{date}</td>
                                <td style={styles.td} className="mono" style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>{rec.filename}</td>
                                <td style={styles.td} className="mono" style={{ fontWeight: 700 }}>{Math.round(rec.pd_probability * 100)}%</td>
                                <td style={styles.td} className="mono">{rec.severity_score ? rec.severity_score.toFixed(1) : 'N/A'}</td>
                                <td style={styles.td}>
                                  <span className={`risk-badge ${badgeClass}`}>{rec.risk_tier || 'Low'}</span>
                                </td>
                                <td style={styles.td} style={{ textAlign: 'right' }}>
                                  <Link to={`/report/${rec.id}`} className="btn btn-secondary btn-sm" style={{ padding: '4px 8px', fontSize: '0.72rem' }}>
                                    Report
                                  </Link>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  header: {
    marginBottom: '24px',
  },
  title: {
    fontSize: '2rem',
    fontWeight: 950,
    background: 'var(--gradient-teal)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    marginBottom: '6px',
  },
  subtitle: {
    fontSize: '0.92rem',
    color: 'var(--text-secondary)',
  },
  mainGrid: {
    display: 'grid',
    gridTemplateColumns: '320px 1fr',
    gap: '24px',
  },
  sidebar: {
    display: 'flex',
    flexDirection: 'column',
  },
  sidebarCard: {
    padding: '20px',
    height: 'fit-content',
  },
  sidebarHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '16px',
    borderBottom: '1px solid var(--glass-border)',
    paddingBottom: '12px',
  },
  addPatientForm: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    background: 'rgba(255,255,255,0.01)',
    border: '1px solid rgba(255,255,255,0.05)',
    borderRadius: 'var(--radius-md)',
    padding: '12px',
    marginBottom: '16px',
  },
  patientList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    maxHeight: '480px',
    overflowY: 'auto',
  },
  patientItem: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 14px',
    borderRadius: 'var(--radius-md)',
    border: '1px solid transparent',
    cursor: 'pointer',
    transition: 'all var(--transition-fast)',
  },
  patientName: {
    fontWeight: 700,
    fontSize: '0.88rem',
    color: 'var(--text-primary)',
  },
  patientMeta: {
    fontSize: '0.75rem',
    color: 'var(--text-muted)',
  },
  patientStatusBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  emptySidebar: {
    textAlign: 'center',
    padding: '32px 0',
    color: 'var(--text-muted)',
    fontSize: '0.85rem',
    fontStyle: 'italic',
  },
  contentArea: {
    minWidth: 0, // Prevents flex charts from overflowing
  },
  emptyContent: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '64px 32px',
    textAlign: 'center',
    height: '320px',
  },
  loaderArea: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '320px',
  },
  patientBanner: {
    padding: '20px 24px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    background: 'var(--gradient-card)',
  },
  bannerMeta: {
    display: 'flex',
    gap: '24px',
  },
  bannerMetaItem: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-end',
  },
  bmLabel: {
    fontSize: '0.7rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
  },
  bmValue: {
    fontSize: '1rem',
    fontWeight: 800,
    color: 'var(--text-primary)',
  },
  trendsSummaryRow: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '24px',
  },
  trendSummaryCard: {
    padding: '16px 20px',
  },
  trendSummaryLabel: {
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  trendRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginTop: '6px',
  },
  trendVal: {
    fontSize: '0.95rem',
    fontWeight: 700,
  },
  chartPanel: {
    padding: '24px',
  },
  panelHeaderRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '12px',
  },
  panelTitle: {
    fontSize: '1.05rem',
    fontWeight: 700,
    color: 'var(--text-primary)',
  },
  panelDesc: {
    fontSize: '0.78rem',
    color: 'var(--text-muted)',
    marginTop: '2px',
  },
  tooltipStyle: {
    background: 'rgba(6, 15, 30, 0.95)',
    border: '1px solid var(--glass-border)',
    borderRadius: 'var(--radius-md)',
    fontSize: '0.8rem',
    color: 'var(--text-primary)',
  },
  progressionDetailGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1.2fr 1fr',
    gap: '16px',
    background: 'rgba(184, 163, 143, 0.12)',
    border: '1px solid rgba(107, 79, 58, 0.22)',
    borderRadius: 'var(--radius-md)',
    padding: '20px',
    marginTop: '16px',
  },
  forecastMetricCard: {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
  },
  forecastLabel: {
    fontSize: '0.75rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
  },
  forecastVal: {
    fontSize: '1.75rem',
    fontWeight: 900,
    marginTop: '4px',
    color: 'var(--text-primary)',
  },
  progressionBadge: {
    fontSize: '1.25rem',
    fontWeight: 800,
    marginTop: '4px',
  },
  insufficientDataBox: {
    display: 'flex',
    gap: '12px',
    background: 'rgba(245, 158, 11, 0.05)',
    border: '1px solid rgba(245, 158, 11, 0.2)',
    borderRadius: 'var(--radius-md)',
    padding: '16px',
    marginTop: '16px',
  },
  experimentalNoticeBox: {
    display: 'flex',
    gap: '12px',
    background: 'rgba(184, 163, 143, 0.12)',
    border: '1px solid rgba(107, 79, 58, 0.24)',
    borderRadius: 'var(--radius-md)',
    padding: '16px',
    marginTop: '16px',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    textAlign: 'left',
  },
  trHead: {
    borderBottom: '1px solid var(--glass-border)',
  },
  th: {
    padding: '10px 12px',
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    fontWeight: 600,
  },
  tr: {
    borderBottom: '1px solid rgba(255,255,255,0.02)',
  },
  td: {
    padding: '12px',
    fontSize: '0.82rem',
  },
};
