import React from 'react';

interface RiskGaugeProps {
  probability: number; // 0 to 1
  confidence?: number; // 0 to 1
  riskTier?: 'low' | 'moderate' | 'high' | 'very_high';
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({ probability, confidence, riskTier }) => {
  const score = Math.round(probability * 100);
  const confidencePercent = confidence ? Math.round(confidence * 100) : null;

  // Determine colors and labels based on risk tier
  let color = 'var(--green-500)';
  let glowColor = 'rgba(16, 185, 129, 0.4)';
  let label = 'Low Risk';
  let badgeClass = 'low';

  const tier = riskTier || (probability < 0.3 ? 'low' : probability < 0.6 ? 'moderate' : probability < 0.85 ? 'high' : 'very_high');

  if (tier === 'moderate') {
    color = 'var(--amber-500)';
    glowColor = 'rgba(245, 158, 11, 0.4)';
    label = 'Moderate Risk';
    badgeClass = 'moderate';
  } else if (tier === 'high') {
    color = 'var(--walnut)';
    glowColor = 'rgba(255, 140, 66, 0.4)';
    label = 'High Risk';
    badgeClass = 'high';
  } else if (tier === 'very_high') {
    color = 'var(--red-500)';
    glowColor = 'rgba(239, 68, 68, 0.4)';
    label = 'Very High Risk';
    badgeClass = 'very_high';
  }

  // SVG calculations for arc
  const radius = 70;
  const strokeWidth = 10;
  const normalizedRadius = radius - strokeWidth / 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  // Arc is only a half-circle (or 3/4 circle). Let's do a 240-degree gauge (leaving bottom open)
  const angleRange = 240;
  const strokeDashoffset = circumference - (score / 100) * (angleRange / 360) * circumference;

  return (
    <div style={styles.gaugeCard}>
      <div style={styles.gaugeContainer}>
        {/* SVG Arc Gauge */}
        <svg height="160" width="160" style={styles.svg}>
          {/* Background track */}
          <circle
            stroke="rgba(255, 255, 255, 0.06)"
            fill="transparent"
            strokeWidth={strokeWidth}
            strokeDasharray={`${(angleRange / 360) * circumference} ${circumference}`}
            style={{
              transform: 'rotate(150deg)',
              transformOrigin: '50% 50%',
            }}
            r={normalizedRadius}
            cx="80"
            cy="80"
          />
          {/* Foreground progress */}
          <circle
            stroke={color}
            fill="transparent"
            strokeWidth={strokeWidth}
            strokeDasharray={`${circumference} ${circumference}`}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            style={{
              transform: 'rotate(150deg)',
              transformOrigin: '50% 50%',
              transition: 'stroke-dashoffset 1s cubic-bezier(0.4, 0, 0.2, 1)',
              filter: `drop-shadow(0 0 6px ${glowColor})`,
            }}
            r={normalizedRadius}
            cx="80"
            cy="80"
          />
        </svg>

        {/* Center overlay texts */}
        <div style={styles.centerText}>
          <div style={styles.percentage}>{score}%</div>
          <div style={styles.percentLabel}>Probability</div>
        </div>
      </div>

      <div style={styles.meta}>
        <span className={`risk-badge ${badgeClass}`}>{label}</span>
        {confidencePercent !== null && (
          <div style={styles.confidence}>
            <span style={styles.confidenceLabel}>Confidence Score</span>
            <span style={styles.confidenceVal}>{confidencePercent}%</span>
          </div>
        )}
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  gaugeCard: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '24px',
    background: 'rgba(11, 26, 46, 0.5)',
    border: '1px solid var(--glass-border)',
    borderRadius: 'var(--radius-lg)',
    width: '100%',
    textAlign: 'center',
  },
  gaugeContainer: {
    position: 'relative',
    height: '160px',
    width: '160px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  svg: {
    position: 'absolute',
    top: 0,
    left: 0,
  },
  centerText: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: '-10px',
  },
  percentage: {
    fontSize: '2.5rem',
    fontWeight: 900,
    color: 'var(--text-primary)',
    lineHeight: 1,
    letterSpacing: '-0.03em',
  },
  percentLabel: {
    fontSize: '0.75rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    marginTop: '4px',
  },
  meta: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '12px',
    marginTop: '12px',
    width: '100%',
  },
  confidence: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    background: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 'var(--radius-sm)',
    padding: '4px 12px',
    fontSize: '0.78rem',
  },
  confidenceLabel: {
    color: 'var(--text-secondary)',
  },
  confidenceVal: {
    color: 'var(--teal-400)',
    fontWeight: 600,
    fontFamily: 'var(--font-mono)',
  },
};
