import axios from 'axios';

// Create an Axios instance with base URL pointing to /api (proxied by Vite)
const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ? `${import.meta.env.VITE_API_BASE_URL}/api` : '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Patient {
  id: string;
  name: string;
  age?: number;
  gender?: string;
  notes?: string;
  created_at: string;
  analysis_count?: number;
  latest_risk_tier?: string;
  latest_pd_probability?: number;
}

export interface Analysis {
  id: string;
  status: 'pending' | 'processing' | 'complete' | 'failed';
  filename: string;
  created_at: string;
  // Calibrated risk assessment
  pd_probability?: number;
  risk_tier?: 'low' | 'moderate' | 'high' | 'very_high';
  confidence?: number;              // numeric 0-1 (distance from 0.5 boundary)
  confidence_category?: 'High' | 'Moderate' | 'Low'; // ensemble agreement
  uncertainty_score?: number;
  uncertainty?: {
    confidence: number;
    uncertainty_score: number;
    warnings: string[];
    ensemble_disagreement: number;
    calibration_confidence: number;
    audio_quality_factor: number;
  };
  // UPDRS severity regression with 95% CI
  severity_score?: number;
  predicted_updrs?: number;
  severity_tier?: 'Healthy' | 'Mild' | 'Moderate' | 'Severe';
  severity_lower_ci?: number;
  severity_upper_ci?: number;
  lower_bound?: number;
  upper_bound?: number;
  // Model & feature outputs
  model_predictions?: Record<string, number>;
  features?: Record<string, number>;
  shap_values?: Record<string, number>;
  feature_importance?: Array<{ feature: string; importance: number }>;
  risk_breakdown?: Record<string, { shap_val: number; value: number; contribution: string; normal_range: string; status: string }>;
  clinical_explanations?: Array<{ label: string; value: number; status: string; clinical_explanation: string; shap_value?: number }>;
  biomarkers?: Array<{ key: string; name: string; value: number; healthy_range: string; status: 'Normal' | 'Borderline' | 'Abnormal'; explanation: string }>;
  quality?: Record<string, any>;
  error_message?: string;
  // Regulatory context
  disclaimer?: string;
}

export interface PatientHistory {
  patient_id: string;
  recordings: Array<{
    id: string;
    date: string;
    filename: string;
    pd_probability: number;
    risk_tier: string;
    severity_score: number;
    severity_tier: string;
    severity_lower_ci?: number;
    severity_upper_ci?: number;
  }>;
  trends: {
    pd_probability: 'stable' | 'increasing' | 'decreasing' | 'insufficient_data';
    severity: 'stable' | 'increasing' | 'decreasing' | 'insufficient_data';
    n_recordings: number;
  };
  /** Experimental — available only with ≥3 sessions in research/experimental/ */
  progression_prediction?: {
    progression_risk: number;
    trend: string;
    trend_confidence: number;
    n_recordings_used: number;
    disclaimer: string;
    error?: string;
  } | null;
}

export interface PatientTrends {
  dates: string[];
  pd_probability: number[];
  severity_score: number[];
  early_detection_score: number[];
  progression_risk: number[];
  fo_mean: number[];
  jitter_local: number[];
  shimmer_local: number[];
  hnr: number[];
  ppe: number[];
}

export interface ModelStatus {
  models: {
    random_forest: boolean;
    xgboost: boolean;
    lightgbm: boolean;
    ensemble: boolean;
    efficientnet_b0: boolean;
    severity: boolean;
    onnx: boolean;
  };
  scaler_ready: boolean;
  any_model_ready: boolean;
  full_pipeline_ready: boolean;
}

export interface EdgeBenchmarkResult {
  profiles: Array<{
    device: string;
    model?: string;
    cpu_cores?: number;
    ram_gb?: number;
    latency_ms: number;
    throughput_samples_per_sec: number;
    model_size_mb: number;
    memory_usage_mb?: number | null;
    compression_ratio?: number | null;
    quantization: string;
    deployable: boolean;
    notes: string;
  }>;
  onnx_available: boolean;
  real_measurements_only?: boolean;
  message?: string | null;
  benchmark_timestamp: string;
}

export interface ModelComparisonRow {
  model: string;
  display_name: string;
  type: 'Classical ML' | 'Deep Learning';
  accuracy?: number;
  roc_auc?: number;
  f1?: number;
  sensitivity?: number;
  specificity?: number;
  precision?: number;
  training_time_sec?: number;
}

export interface ModelComparisonResponse {
  comparison_table: ModelComparisonRow[];
  best_model: string | null;
  metrics_available: boolean;
}

export const api = {
  // Patients CRUD
  getPatients: () => client.get<Patient[]>('/patients/').then((res) => res.data),
  getPatient: (id: string) => client.get<Patient>(`/patients/${id}`).then((res) => res.data),
  createPatient: (data: Omit<Patient, 'id' | 'created_at'>) => client.post<Patient>('/patients/', data).then((res) => res.data),
  updatePatient: (id: string, data: Partial<Omit<Patient, 'id' | 'created_at'>>) => client.put<Patient>(`/patients/${id}`, data).then((res) => res.data),
  deletePatient: (id: string) => client.delete(`/patients/${id}`).then((res) => res.data),
  getPatientHistory: (id: string) => client.get<PatientHistory>(`/patients/${id}/history`).then((res) => res.data),
  getPatientTrends: (id: string) => client.get<PatientTrends>(`/patients/${id}/trends`).then((res) => res.data),

  // Analyses
  uploadAudio: (file: File, patientId?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (patientId) {
      formData.append('patient_id', patientId);
    }
    return client.post<Analysis>('/analysis/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }).then((res) => res.data);
  },
  getAnalysis: (id: string) => client.get<Analysis>(`/analysis/${id}`).then((res) => res.data),
  getExplanation: (id: string) => client.get<any>(`/analysis/${id}/explain`).then((res) => res.data),
  deleteAnalysis: (id: string) => client.delete(`/analysis/${id}`).then((res) => res.data),
  listAnalyses: (opts: { skip?: number; limit?: number; patientId?: string } | number = {}, limit = 20, patientId?: string) => {
    // Support both object and positional-arg call signatures
    const skip = typeof opts === 'object' ? (opts.skip ?? 0) : opts;
    const lim  = typeof opts === 'object' ? (opts.limit ?? 20) : limit;
    const pid  = typeof opts === 'object' ? opts.patientId : patientId;
    return client.get<{ items: Analysis[]; total: number }>('/analysis/', {
      params: { skip, limit: lim, patient_id: pid },
    }).then((res) => res.data);
  },

  // Model Status & Comparison
  getModelStatus: () => client.get<ModelStatus>('/models/status').then((res) => res.data),
  getModelMetrics: () => client.get<any>('/models/metrics').then((res) => res.data),
  getModelComparison: () => client.get<ModelComparisonResponse>('/models/comparison').then((res) => res.data),
  getModelRegistry: () => client.get<any>('/models/registry').then((res) => res.data),
  triggerTraining: () => client.post<any>('/models/train').then((res) => res.data),

  // Edge benchmarks
  getEdgeBenchmarks: () => client.get<EdgeBenchmarkResult>('/models/benchmarks').then((res) => res.data),
};
