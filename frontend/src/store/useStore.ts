import { create } from 'zustand';
import { api, type Patient, type Analysis, type ModelStatus } from '../api/client';

interface State {
  patients: Patient[];
  selectedPatientId: string | null;
  recentAnalyses: Analysis[];
  modelStatus: ModelStatus | null;
  loading: boolean;
  error: string | null;

  // Actions
  fetchPatients: () => Promise<void>;
  setSelectedPatientId: (id: string | null) => void;
  createPatient: (data: Omit<Patient, 'id' | 'created_at'>) => Promise<Patient>;
  fetchRecentAnalyses: (patientId?: string) => Promise<void>;
  fetchModelStatus: () => Promise<void>;
}

export const useStore = create<State>((set, get) => ({
  patients: [],
  selectedPatientId: null,
  recentAnalyses: [],
  modelStatus: null,
  loading: false,
  error: null,

  fetchPatients: async () => {
    set({ loading: true, error: null });
    try {
      const data = await api.getPatients();
      set({ patients: data, loading: false });
    } catch (err: any) {
      set({ error: err.message || 'Failed to fetch patients', loading: false });
    }
  },

  setSelectedPatientId: (id) => {
    set({ selectedPatientId: id });
  },

  createPatient: async (data) => {
    set({ loading: true, error: null });
    try {
      const newPatient = await api.createPatient(data);
      const currentPatients = get().patients;
      set({ 
        patients: [newPatient, ...currentPatients], 
        selectedPatientId: newPatient.id,
        loading: false 
      });
      return newPatient;
    } catch (err: any) {
      set({ error: err.message || 'Failed to create patient', loading: false });
      throw err;
    }
  },

  fetchRecentAnalyses: async (patientId) => {
    set({ loading: true, error: null });
    try {
      const data = await api.listAnalyses(0, 20, patientId);
      set({ recentAnalyses: data.items, loading: false });
    } catch (err: any) {
      set({ error: err.message || 'Failed to fetch recent analyses', loading: false });
    }
  },

  fetchModelStatus: async () => {
    try {
      const status = await api.getModelStatus();
      set({ modelStatus: status });
    } catch (err: any) {
      console.error('Failed to fetch model status:', err);
    }
  },
}));
