import React, { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { api, type Analysis, type Patient } from '../api/client';
import { RiskGauge } from '../components/RiskGauge';
import { AudioWaveform } from '../components/AudioWaveform';
import {
  FileText, BarChart2, ShieldAlert, Award,
  Activity, ArrowRight, User, Calendar,
  AlertCircle, CheckCircle, FileDown, Trash2,
  AlertTriangle, Info, TrendingUp
} from 'lucide-react';

export const Report: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [patient, setPatient] = useState<Patient | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    const fetchReportData = async () => {
      try {
        setLoading(true);
        const data = await api.getAnalysis(id);
        setAnalysis(data);

        if (data.patient_id) {
          const p = await api.getPatient(data.patient_id);
          setPatient(p);
        }
        setLoading(false);
      } catch (err: any) {
        console.error("Error fetching report data", err);
        setError("Failed to retrieve the analysis report. Please verify the ID.");
        setLoading(false);
      }
    };
    fetchReportData();
  }, [id]);

  const handleDelete = async () => {
    if (!id) return;
    if (window.confirm("Are you sure you want to permanently delete this report and audio recording?")) {
      try {
        await api.deleteAnalysis(id);
        navigate('/');
      } catch (err) {
        console.error(err);
        alert("Failed to delete record.");
      }
    }
  };

  if (loading) {
    return (
      <div style={styles.loadingContainer}>
        <div className="spinner" />
        <p style={{ marginTop: '16px', color: 'var(--text-secondary)' }}>Loading screening parameters...</p>
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="page-layout fade-in" style={styles.errorContainer}>
        <div className="glass-card" style={styles.errorCard}>
          <AlertCircle size={40} style={{ color: 'var(--red-400)' }} />
          <h2 style={{ fontSize: '1.2rem', marginTop: '12px' }}>Assessment Retrieve Error</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', margin: '8px 0 16px' }}>{error || 'Record not found.'}</p>
          <Link to="/" className="btn btn-primary">Return Home</Link>
        </div>
      </div>
    );
  }

  const dateStr = new Date(analysis.created_at).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  // Severity indicator color mapping
  const getSeverityColor = (tier?: string) => {
    if (tier === 'mild') return 'var(--green-400)';
    if (tier === 'moderate') return 'var(--amber-400)';
    if (tier === 'severe') return 'var(--red-400)';
    return 'var(--text-muted)';
  };

  return (
    <div className="page-layout fade-in">
      {/* ── Clinical Disclaimer Banner ── */}
      <div style={styles.disclaimerBanner}>
        <AlertTriangle size={14} style={{ color: 'var(--amber-400)', flexShrink: 0 }} />
        <span style={styles.disclaimerText}>
          <strong>Research Tool:</strong> This system is intended for research and screening support
          purposes only and is <strong>not a diagnostic medical device</strong>. Always consult a
          qualified neurologist for clinical assessment.
        </span>
      </div>

      {/* Top action row */}
      <div style={styles.actionRow}>
        <div style={styles.titleArea}>
          <FileText size={22} style={{ color: 'var(--teal-500)' }} />
          <span style={{ fontSize: '1.15rem', fontWeight: 800 }}>Screening Assessment Report</span>
        </div>
        <div style={{ display: 'flex', gap: '12px' }}>
          <button onClick={handleDelete} className="btn btn-ghost btn-sm" style={{ borderColor: 'rgba(58, 42, 32, 0.22)', color: 'var(--red-400)' }}>
            <Trash2 size={14} />
            <span>Delete Record</span>
          </button>
          <Link to={`/explain/${analysis.id}`} className="btn btn-primary btn-sm">
            <span>Explore SHAP Explainability</span>
            <ArrowRight size={14} />
          </Link>
        </div>
      </div>

      <div style={styles.grid}>
        {/* Left Column */}
        <div style={styles.leftCol}>
          {/* Patient Details Card */}
          <div className="glass-card" style={styles.card}>
            <h3 style={styles.cardTitle}>
              <User size={18} style={{ color: 'var(--teal-400)' }} />
              <span>Subject Profile</span>
            </h3>
            <div style={styles.metaGrid}>
              <div style={styles.metaItem}>
                <span style={styles.metaLabel}>Name</span>
                <span style={styles.metaValue}>{patient?.name || 'Anonymized Evaluation'}</span>
              </div>
              <div style={styles.metaItem}>
                <span style={styles.metaLabel}>Biological Age</span>
                <span style={styles.metaValue}>{patient?.age ? `${patient.age} yrs` : 'N/A'}</span>
              </div>
              <div style={styles.metaItem}>
                <span style={styles.metaLabel}>Gender</span>
                <span style={styles.metaValue}>{patient?.gender || 'N/A'}</span>
              </div>
              <div style={styles.metaItem}>
                <span style={styles.metaLabel}>Evaluation Date</span>
                <span style={styles.metaValue}>{dateStr}</span>
              </div>
            </div>
            {patient?.notes && (
              <div style={styles.notesBox}>
                <strong style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>Clinical Observations:</strong>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', fontStyle: 'italic', lineHeight: 1.4 }}>{patient.notes}</p>
              </div>
            )}
          </div>

          {/* Primary PD Risk Gauge */}
          <div style={styles.gaugeBlock}>
            <RiskGauge
              probability={analysis.pd_probability || 0}
              confidence={analysis.confidence}
              riskTier={analysis.risk_tier}
            />
          </div>

          {/* Audio recording & signal details */}
          <div className="glass-card" style={styles.card}>
            <h3 style={styles.cardTitle}>Acoustic Playback</h3>
            {analysis.id && <AudioWaveform src={`/uploads/${analysis.id}_${analysis.filename}`} />}
            
            <div style={styles.signalQuality}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Audio Signal Integrity:</span>
              <div style={styles.qualityRow}>
                <div style={styles.qualityItem}>
                  <CheckCircle size={14} style={{ color: 'var(--green-400)' }} />
                  <span>No clipping detected</span>
                </div>
                <div style={styles.qualityItem}>
                  <CheckCircle size={14} style={{ color: 'var(--green-400)' }} />
                  <span>SNR valid (&gt;18dB)</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div style={styles.rightCol}>
          {/* Clinical Severity UPDRS Regression */}
          <div className="glass-card" style={styles.card}>
            <h3 style={styles.cardTitle}>
              <Activity size={18} style={{ color: 'var(--teal-400)' }} />
              <span>Unified Parkinson's Disease Rating (UPDRS)</span>
            </h3>
            <p style={styles.cardDesc}>
              Tabular regression assessment predicting Unified Parkinson's Disease Rating Scale (UPDRS III - motor subsection) based on tremor severity and instability levels.
            </p>

            <div style={styles.severitySection}>
              <div style={styles.severityMetric}>
                <span style={{ ...styles.severityValue, color: getSeverityColor(analysis.severity_tier) }}>
                  {analysis.severity_score !== undefined && analysis.severity_score !== null ? analysis.severity_score.toFixed(1) : 'N/A'}
                </span>
                <span style={styles.severityLabel}>Predicted UPDRS-III Score</span>
              </div>
              <div style={styles.severityText}>
                <span style={styles.severityTierLabel}>Severity Classification</span>
                <span className="mono" style={{ ...styles.severityTierVal, color: getSeverityColor(analysis.severity_tier) }}>
                  {(analysis.severity_tier || 'Unknown').toUpperCase()}
                </span>
              </div>
            </div>

            {/* UPDRS Range Slider display */}
            <div style={styles.sliderContainer}>
              <div style={styles.sliderTicks}>
                <span>0 (Normal)</span>
                <span>54 (Moderate)</span>
                <span>108 (Severe)</span>
              </div>
              <div style={styles.sliderBar}>
                <div
                  style={{
                    ...styles.sliderThumb,
                    left: `${(analysis.severity_score || 0) / 108 * 100}%`,
                    background: getSeverityColor(analysis.severity_tier)
                  }}
                />
              </div>
            </div>
          </div>

          {/* Model Consensus Grid */}
          <div className="glass-card" style={styles.card}>
            <h3 style={styles.cardTitle}>
              <Award size={18} style={{ color: 'var(--teal-400)' }} />
              <span>Multi-Model Screening Consensus</span>
            </h3>
            <p style={styles.cardDesc}>Individual calibrated classifier outputs contributing to the weighted ensemble screening result.</p>

            {/* Confidence Category Badge */}
            {analysis.confidence_category && (
              <div style={styles.confidenceBadgeRow}>
                <span style={styles.confidenceBadgeLabel}>Ensemble Agreement:</span>
                <span style={{
                  ...styles.confidenceBadge,
                  color: analysis.confidence_category === 'High' ? 'var(--green-400)'
                       : analysis.confidence_category === 'Moderate' ? 'var(--amber-400)'
                       : 'var(--red-400)',
                  borderColor: analysis.confidence_category === 'High' ? 'rgba(140,115,95,0.32)'
                             : analysis.confidence_category === 'Moderate' ? 'rgba(107,79,58,0.32)'
                             : 'rgba(58,42,32,0.32)',
                }}>
                  {analysis.confidence_category} Confidence
                </span>
              </div>
            )}

            <div style={styles.consensusList}>
              {analysis.model_predictions && Object.entries(analysis.model_predictions).map(([model, prob]) => {
                const percent = Math.round(prob * 100);
                const isPD = prob >= 0.5;
                const modelLabel = model.toUpperCase().replace('_', ' ');

                return (
                  <div key={model} style={styles.consensusRow}>
                    <div style={styles.consensusMeta}>
                      <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>{modelLabel}</span>
                      <span style={{ color: isPD ? 'var(--red-400)' : 'var(--green-400)', fontSize: '0.85rem', fontWeight: 700 }}>
                        {percent}% {isPD ? 'PD Risk' : 'Healthy'}
                      </span>
                    </div>
                    <div className="progress-bar" style={{ height: '4px' }}>
                      <div
                        className="progress-fill"
                        style={{
                          width: `${percent}%`,
                          background: isPD ? 'var(--red-500)' : 'var(--green-500)',
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* UPDRS Confidence Interval Panel */}
          <div style={styles.indicatorGrid}>
            <div className="glass-card" style={styles.indicatorCard}>
              <span style={styles.indicatorTitle}>95% Confidence Interval</span>
              <div style={styles.ciRow}>
                <span style={{ ...styles.indicatorVal, fontSize: '1.2rem', color: 'var(--text-muted)' }}>
                  {analysis.severity_lower_ci !== undefined && analysis.severity_lower_ci !== null
                    ? analysis.severity_lower_ci.toFixed(1) : '—'}
                </span>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', alignSelf: 'center' }}>–</span>
                <span style={{ ...styles.indicatorVal, fontSize: '1.2rem', color: 'var(--text-muted)' }}>
                  {analysis.severity_upper_ci !== undefined && analysis.severity_upper_ci !== null
                    ? analysis.severity_upper_ci.toFixed(1) : '—'}
                </span>
              </div>
              <p style={styles.indicatorDesc}>UPDRS-III predicted range (95% CI from validation residuals). Use this range to interpret score uncertainty.</p>
            </div>

            <div className="glass-card" style={styles.indicatorCard}>
              <span style={styles.indicatorTitle}>Ensemble Confidence</span>
              <span style={styles.indicatorVal}>
                {analysis.confidence !== undefined && analysis.confidence !== null
                  ? `${Math.round(analysis.confidence * 100)}%`
                  : 'N/A'}
              </span>
              <p style={styles.indicatorDesc}>Model decisiveness score — distance of the ensemble probability from the 0.5 decision boundary, scaled 0–100%.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  disclaimerBanner: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '10px',
    background: 'rgba(184, 163, 143, 0.12)',
    border: '1px solid rgba(107, 79, 58, 0.22)',
    borderRadius: 'var(--radius-md)',
    padding: '10px 16px',
    marginBottom: '20px',
  },
  disclaimerText: {
    fontSize: '0.78rem',
    color: 'var(--amber-300)',
    lineHeight: 1.45,
  },
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
  actionRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '24px',
  },
  titleArea: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '1.2fr 1.8fr',
    gap: '24px',
  },
  leftCol: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
  },
  rightCol: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
  },
  card: {
    padding: '24px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  cardTitle: {
    fontSize: '1.05rem',
    fontWeight: 700,
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  cardDesc: {
    fontSize: '0.82rem',
    color: 'var(--text-muted)',
    lineHeight: 1.45,
  },
  metaGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '16px',
  },
  metaItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  metaLabel: {
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  metaValue: {
    fontSize: '0.9rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  notesBox: {
    background: 'rgba(255, 255, 255, 0.02)',
    border: '1px solid rgba(255, 255, 255, 0.05)',
    borderRadius: 'var(--radius-sm)',
    padding: '12px',
  },
  gaugeBlock: {
    width: '100%',
  },
  signalQuality: {
    marginTop: '8px',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  qualityRow: {
    display: 'flex',
    gap: '20px',
  },
  qualityItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '0.8rem',
    color: 'var(--text-secondary)',
  },
  severitySection: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    background: 'rgba(6, 15, 30, 0.6)',
    padding: '16px 20px',
    borderRadius: 'var(--radius-md)',
    border: '1px solid rgba(0, 212, 170, 0.05)',
  },
  severityMetric: {
    display: 'flex',
    flexDirection: 'column',
  },
  severityValue: {
    fontSize: '2rem',
    fontWeight: 900,
    lineHeight: 1,
  },
  severityLabel: {
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
    marginTop: '4px',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  severityText: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-end',
  },
  severityTierLabel: {
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  severityTierVal: {
    fontSize: '1.1rem',
    fontWeight: 800,
    marginTop: '2px',
  },
  sliderContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    width: '100%',
  },
  sliderTicks: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
  },
  sliderBar: {
    width: '100%',
    height: '6px',
    background: 'rgba(255, 255, 255, 0.08)',
    borderRadius: 'var(--radius-full)',
    position: 'relative',
  },
  sliderThumb: {
    width: '12px',
    height: '12px',
    borderRadius: '50%',
    position: 'absolute',
    top: '50%',
    transform: 'translate(-50%, -50%)',
    boxShadow: '0 0 10px rgba(0, 212, 170, 0.4)',
    transition: 'left 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
  },
  consensusList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  consensusRow: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  consensusMeta: {
    display: 'flex',
    justifyContent: 'space-between',
  },
  indicatorGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '24px',
  },
  indicatorCard: {
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  ciRow: {
    display: 'flex',
    gap: '8px',
    alignItems: 'baseline',
    margin: '6px 0 2px',
  },
  confidenceBadgeRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '4px',
  },
  confidenceBadgeLabel: {
    fontSize: '0.78rem',
    color: 'var(--text-muted)',
  },
  confidenceBadge: {
    fontSize: '0.78rem',
    fontWeight: 700,
    padding: '2px 10px',
    borderRadius: 'var(--radius-full)',
    border: '1px solid',
  },
  indicatorTitle: {
    fontSize: '0.75rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    fontWeight: 600,
  },
  indicatorVal: {
    fontSize: '1.6rem',
    fontWeight: 800,
    color: 'var(--text-primary)',
    margin: '4px 0',
  },
  indicatorDesc: {
    fontSize: '0.75rem',
    color: 'var(--text-secondary)',
    lineHeight: 1.35,
  },
};
