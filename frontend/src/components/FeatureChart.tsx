import React from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  CartesianGrid,
} from 'recharts';

interface RadarDataPoint {
  group: string;
  risk: number;
  level: string;
}

interface ShapDataPoint {
  feature: string;
  display_name: string;
  shap_contribution: number;
  value?: number;
  direction: 'risk' | 'protective';
  status: string;
}

interface FeatureChartProps {
  radarData?: RadarDataPoint[];
  shapData?: ShapDataPoint[];
  type: 'radar' | 'shap';
}

export const FeatureChart: React.FC<FeatureChartProps> = ({ radarData = [], shapData = [], type }) => {
  if (type === 'radar') {
    if (radarData.length === 0) {
      return <div style={styles.empty}>No biomarker group data available.</div>;
    }

    return (
      <div style={styles.chartWrapper}>
        <ResponsiveContainer width="100%" height={320}>
          <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
            <PolarGrid stroke="rgba(0, 212, 170, 0.15)" />
            <PolarAngleAxis 
              dataKey="group" 
              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
            />
            <PolarRadiusAxis 
              angle={30} 
              domain={[0, 100]} 
              tick={{ fill: 'var(--text-muted)', fontSize: 9 }}
            />
            <Radar
              name="Biomarker Risk"
              dataKey="risk"
              stroke="var(--teal-500)"
              fill="rgba(0, 212, 170, 0.2)"
              fillOpacity={0.6}
            />
            <Tooltip
              contentStyle={styles.tooltip}
              itemStyle={{ color: 'var(--text-primary)' }}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  // Bar chart of SHAP values
  if (type === 'shap') {
    if (shapData.length === 0) {
      return <div style={styles.empty}>No explainability data available.</div>;
    }

    // Sort features by absolute contribution, limit to top 10 for readability
    const sortedShap = [...shapData]
      .sort((a, b) => Math.abs(b.shap_contribution) - Math.abs(a.shap_contribution))
      .slice(0, 10);

    return (
      <div style={styles.chartWrapper}>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart
            data={sortedShap}
            layout="vertical"
            margin={{ top: 10, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis 
              type="number" 
              tick={{ fill: 'var(--text-secondary)', fontSize: 10 }}
              stroke="rgba(255,255,255,0.1)"
              label={{ value: 'SHAP Value (Prediction Contribution)', position: 'insideBottom', offset: -5, fill: 'var(--text-muted)', fontSize: 11 }}
            />
            <YAxis
              type="category"
              dataKey="display_name"
              tick={{ fill: 'var(--text-primary)', fontSize: 11 }}
              stroke="rgba(255,255,255,0.1)"
              width={150}
            />
            <Tooltip
              cursor={{ fill: 'rgba(255,255,255,0.02)' }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data: ShapDataPoint = payload[0].payload;
                  const valueStr = data.value !== undefined && data.value !== null ? data.value.toFixed(4) : 'N/A';
                  return (
                    <div style={styles.customTooltip}>
                      <p style={styles.tooltipTitle}>{data.display_name}</p>
                      <p style={styles.tooltipRow}>
                        <span>Measured Value:</span> <strong>{valueStr}</strong>
                      </p>
                      <p style={styles.tooltipRow}>
                        <span>Impact Score:</span>{' '}
                        <strong style={{ color: data.shap_contribution > 0 ? 'var(--red-400)' : 'var(--green-400)' }}>
                          {data.shap_contribution > 0 ? '+' : ''}{data.shap_contribution.toFixed(4)}
                        </strong>
                      </p>
                      <p style={styles.tooltipRow}>
                        <span>Range Status:</span>{' '}
                        <span style={styles.statusBadge(data.status)}>{data.status}</span>
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="shap_contribution">
              {sortedShap.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.shap_contribution > 0 ? 'var(--red-500)' : 'var(--green-500)'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  return null;
};

const styles = {
  chartWrapper: {
    width: '100%',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
  empty: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: '200px',
    color: 'var(--text-muted)',
    fontSize: '0.9rem',
    fontStyle: 'italic',
  },
  tooltip: {
    background: 'rgba(6, 15, 30, 0.95)',
    border: '1px solid var(--glass-border)',
    borderRadius: 'var(--radius-md)',
    padding: '8px 12px',
    fontSize: '0.85rem',
  },
  customTooltip: {
    background: 'rgba(6, 15, 30, 0.96)',
    border: '1px solid var(--glass-border)',
    borderRadius: 'var(--radius-md)',
    padding: '12px',
    boxShadow: 'var(--shadow-md)',
    color: 'var(--text-primary)',
    fontSize: '0.85rem',
  },
  tooltipTitle: {
    fontWeight: 700,
    marginBottom: '8px',
    borderBottom: '1px solid rgba(255,255,255,0.08)',
    paddingBottom: '4px',
    color: 'var(--teal-400)',
  },
  tooltipRow: {
    margin: '4px 0',
    display: 'flex',
    justifyContent: 'space-between',
    gap: '16px',
  },
  statusBadge: (status: string) => {
    let color = 'var(--text-muted)';
    if (status === 'normal') color = 'var(--green-400)';
    if (status === 'borderline') color = 'var(--amber-400)';
    if (status === 'abnormal') color = 'var(--red-400)';
    return {
      color,
      fontWeight: 600,
      textTransform: 'uppercase' as const,
      fontSize: '0.75rem',
    };
  },
};
