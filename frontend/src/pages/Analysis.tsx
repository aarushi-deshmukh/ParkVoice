import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api, type Analysis } from '../api/client';
import { Loader2, CheckCircle2, AlertCircle, Cpu, FileAudio, BarChart } from 'lucide-react';

interface PipelineStep {
  id: string;
  label: string;
  description: string;
}

const PIPELINE_STEPS: PipelineStep[] = [
  { id: 'upload', label: 'Audio Signal Reception', description: 'Trimming silence, resampling to 22,050Hz, and validation checks.' },
  { id: 'preprocess', label: 'Spectral Noise Reduction', description: 'Applying spectral gating to filter out ambient background noise.' },
  { id: 'biomarkers', label: 'Acoustic Biomarker Extraction', description: 'Calculating MFCC coefficients, Pitch (F0), Jitter, and Shimmer.' },
  { id: 'complexity', label: 'Nonlinear Signal Complexity', description: 'Computing signal entropy, RPDE, DFA, and stability metrics.' },
  { id: 'inference', label: 'Ensemble ML Inference', description: 'Evaluating via RF, XGBoost, LightGBM, and Deep CNN architectures.' },
  { id: 'severity', label: 'Severity Classification', description: 'Predicting UPDRS motor scores and temporal risk progression.' },
  { id: 'explain', label: 'SHAP Explainability Calculation', description: 'Generating game-theoretic local feature contributions.' },
];

export const AnalysisPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [pollCount, setPollCount] = useState(0);

  // Poll analysis endpoint
  useEffect(() => {
    if (!id) return;

    let active = true;
    let timer: number;

    const poll = async () => {
      try {
        const data = await api.getAnalysis(id);
        if (!active) return;
        setAnalysis(data);
        setPollCount(prev => prev + 1);

        if (data.status === 'complete') {
          // Finish steps and navigate to report
          setCurrentStepIdx(PIPELINE_STEPS.length);
          setTimeout(() => {
            navigate(`/report/${id}`);
          }, 1200);
        } else if (data.status === 'failed') {
          // Stop polling
        } else {
          // Increment dummy visual steps over time to match active backend processing
          setCurrentStepIdx(prev => Math.min(prev + 1, PIPELINE_STEPS.length - 2));
          timer = window.setTimeout(poll, 1500);
        }
      } catch (err) {
        console.error("Error polling analysis", err);
        timer = window.setTimeout(poll, 3000);
      }
    };

    poll();

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [id, navigate]);

  if (!analysis) {
    return (
      <div style={styles.loadingContainer}>
        <Loader2 size={36} className="spinner" />
        <p style={{ marginTop: '16px', color: 'var(--text-secondary)' }}>Initializing clinical workspace...</p>
      </div>
    );
  }

  const isFailed = analysis.status === 'failed';
  const isComplete = analysis.status === 'complete';

  return (
    <div className="page-layout fade-in" style={styles.container}>
      <div className="glass-card" style={styles.card}>
        {/* Card Header */}
        <div style={styles.header}>
          <div style={styles.iconWrap}>
            <Cpu size={24} style={{ color: 'var(--teal-500)' }} />
          </div>
          <div style={styles.headerText}>
            <h1 style={styles.title}>Acoustic Analysis Processing</h1>
            <p style={styles.subtitle}>ID: <span className="mono">{analysis.id}</span> | Source: <span className="mono">{analysis.filename}</span></p>
          </div>
        </div>

        {/* Global progress indicator */}
        <div style={styles.progressSection}>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: isFailed
                  ? '100%'
                  : `${(currentStepIdx / PIPELINE_STEPS.length) * 100}%`,
                background: isFailed ? 'var(--red-500)' : 'var(--gradient-teal)',
              }}
            />
          </div>
          <div style={styles.progressLabelRow}>
            <span style={{ color: 'var(--text-secondary)' }}>
              {isFailed ? 'Processing Aborted' : isComplete ? 'Analysis Finished' : 'Extracting Voice Biomarkers...'}
            </span>
            <span style={{ fontFamily: 'var(--font-mono)' }}>
              {isFailed ? 'Error' : `${Math.round((currentStepIdx / PIPELINE_STEPS.length) * 100)}%`}
            </span>
          </div>
        </div>

        <div className="divider" style={{ margin: '8px 0' }} />

        {/* Pipeline Step Checklist */}
        <div style={styles.stepsList}>
          {PIPELINE_STEPS.map((step, idx) => {
            const stepState =
              isFailed && idx >= currentStepIdx
                ? 'failed'
                : idx < currentStepIdx
                ? 'done'
                : idx === currentStepIdx
                ? 'active'
                : 'pending';

            return (
              <div
                key={step.id}
                style={{
                  ...styles.stepRow,
                  opacity: stepState === 'pending' ? 0.35 : 1,
                  borderLeft: stepState === 'active' ? '3px solid var(--teal-500)' : '3px solid transparent',
                  paddingLeft: stepState === 'active' ? '13px' : '16px'
                }}
              >
                <div style={styles.stepIndicator}>
                  {stepState === 'done' && <CheckCircle2 size={18} style={{ color: 'var(--teal-400)' }} />}
                  {stepState === 'active' && <Loader2 size={18} className="spinner" style={{ color: 'var(--teal-500)' }} />}
                  {stepState === 'pending' && <div style={styles.pendingDot} />}
                  {stepState === 'failed' && <AlertCircle size={18} style={{ color: 'var(--red-400)' }} />}
                </div>

                <div style={{ flex: 1 }}>
                  <h4
                    style={{
                      ...styles.stepTitle,
                      color: stepState === 'active' ? 'var(--teal-400)' : 'var(--text-primary)'
                    }}
                  >
                    {step.label}
                  </h4>
                  <p style={styles.stepDesc}>{step.description}</p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Error message card if pipeline failed */}
        {isFailed && (
          <div style={styles.errorBox}>
            <div style={{ display: 'flex', gap: '12px' }}>
              <AlertCircle size={20} style={{ color: 'var(--red-400)', flexShrink: 0 }} />
              <div>
                <h4 style={{ color: 'var(--red-400)', fontWeight: 700, fontSize: '0.92rem' }}>Pipeline Exception Encountered</h4>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginTop: '4px' }}>
                  {analysis.error_message || 'An unspecified error occurred during audio feature extraction. Please confirm your voice file contains valid speech phonations and is not corrupted.'}
                </p>
              </div>
            </div>
            <button onClick={() => navigate('/')} className="btn btn-ghost btn-sm" style={{ marginTop: '16px', width: 'fit-content' }}>
              Return to Upload
            </button>
          </div>
        )}
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
  container: {
    maxWidth: '720px',
  },
  card: {
    padding: '32px',
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
  },
  header: {
    display: 'flex',
    gap: '16px',
    alignItems: 'center',
  },
  iconWrap: {
    width: '48px',
    height: '48px',
    borderRadius: 'var(--radius-md)',
    background: 'rgba(0, 212, 170, 0.1)',
    border: '1px solid rgba(0, 212, 170, 0.2)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerText: {
    display: 'flex',
    flexDirection: 'column',
  },
  title: {
    fontSize: '1.4rem',
    fontWeight: 800,
    color: 'var(--text-primary)',
  },
  subtitle: {
    fontSize: '0.82rem',
    color: 'var(--text-muted)',
    marginTop: '2px',
  },
  progressSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  progressLabelRow: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '0.8rem',
    color: 'var(--text-secondary)',
  },
  stepsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  stepRow: {
    display: 'flex',
    gap: '16px',
    alignItems: 'flex-start',
    padding: '12px 16px',
    borderRadius: 'var(--radius-md)',
    background: 'rgba(255, 255, 255, 0.01)',
    border: '1px solid rgba(255, 255, 255, 0.02)',
    transition: 'all 0.3s ease',
  },
  stepIndicator: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '20px',
    height: '24px',
  },
  pendingDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    background: 'rgba(255, 255, 255, 0.15)',
  },
  stepTitle: {
    fontSize: '0.92rem',
    fontWeight: 700,
    marginBottom: '2px',
  },
  stepDesc: {
    fontSize: '0.8rem',
    color: 'var(--text-secondary)',
    lineHeight: 1.4,
  },
  errorBox: {
    display: 'flex',
    flexDirection: 'column',
    background: 'rgba(239, 68, 68, 0.08)',
    border: '1px solid rgba(239, 68, 68, 0.2)',
    borderRadius: 'var(--radius-lg)',
    padding: '20px',
    marginTop: '8px',
  },
};
