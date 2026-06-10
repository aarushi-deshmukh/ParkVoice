import React, { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { useStore } from '../store/useStore';
import { Activity, Mic, History, User, LayoutDashboard, Dna, Cpu, Upload, UserPlus, X } from 'lucide-react';

export const Navbar: React.FC = () => {
  const { patients, selectedPatientId, setSelectedPatientId, fetchPatients, createPatient } = useStore();
  const [showAddPatient, setShowAddPatient] = useState(false);
  const [newPatient, setNewPatient] = useState({ name: '', age: '', gender: 'Male', notes: '' });

  useEffect(() => {
    fetchPatients();
  }, [fetchPatients]);

  const handlePatientSelect = (value: string) => {
    if (value === '__add_patient__') {
      setShowAddPatient(true);
      return;
    }
    setSelectedPatientId(value || null);
  };

  const handleCreatePatient = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!newPatient.name.trim()) return;
    const created = await createPatient({
      name: newPatient.name.trim(),
      age: newPatient.age ? Number(newPatient.age) : undefined,
      gender: newPatient.gender,
      notes: newPatient.notes.trim() || undefined,
    });
    setSelectedPatientId(created.id);
    setNewPatient({ name: '', age: '', gender: 'Male', notes: '' });
    setShowAddPatient(false);
  };

  return (
    <header style={styles.header}>
      <div className="container" style={styles.navContainer}>
        <NavLink to="/" style={styles.logo}>
          <div style={styles.logoIcon}>
            <Activity size={20} />
          </div>
          <span style={styles.logoText}>ParkVoice <span style={{ color: 'var(--walnut)' }}>AI</span></span>
        </NavLink>

        <nav style={styles.navLinks}>
          <NavLink
            to="/"
            style={({ isActive }) => ({ ...styles.link, ...(isActive ? styles.activeLink : {}) })}
            end
          >
            <LayoutDashboard size={16} />
            <span>Overview</span>
          </NavLink>
          <NavLink
            to="/home"
            style={({ isActive }) => ({ ...styles.link, ...(isActive ? styles.activeLink : {}) })}
          >
            <Upload size={16} />
            <span>Upload Audio</span>
          </NavLink>
          <NavLink
            to="/record"
            style={({ isActive }) => ({ ...styles.link, ...(isActive ? styles.activeLink : {}) })}
          >
            <Mic size={16} />
            <span>Record Voice</span>
          </NavLink>
          <NavLink
            to="/history"
            style={({ isActive }) => ({ ...styles.link, ...(isActive ? styles.activeLink : {}) })}
          >
            <History size={16} />
            <span>History</span>
          </NavLink>
          <NavLink
            to="/biomarkers"
            style={({ isActive }) => ({ ...styles.link, ...(isActive ? styles.activeLink : {}) })}
          >
            <Dna size={16} />
            <span>Biomarkers</span>
          </NavLink>
          <NavLink
            to="/benchmarks"
            style={({ isActive }) => ({ ...styles.link, ...(isActive ? styles.activeLink : {}) })}
          >
            <Cpu size={16} />
            <span>Benchmarks</span>
          </NavLink>
        </nav>

        <div style={styles.patientSelector}>
          <User size={16} style={{ color: 'var(--walnut)' }} />
          <select
            value={selectedPatientId || ''}
            onChange={(e) => handlePatientSelect(e.target.value)}
            style={styles.select}
          >
            <option value="" style={styles.option}>-- Select Patient --</option>
            {patients.map((p) => (
              <option key={p.id} value={p.id} style={styles.option}>
                {p.name} (Age {p.age || 'N/A'})
              </option>
            ))}
            <option value="__add_patient__" style={styles.option}>+ Add patient</option>
          </select>
        </div>
      </div>

      {showAddPatient && (
        <div style={styles.modalBackdrop}>
          <form style={styles.modal} onSubmit={handleCreatePatient}>
            <div style={styles.modalHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <UserPlus size={18} />
                <strong>Add Patient</strong>
              </div>
              <button type="button" style={styles.iconButton} onClick={() => setShowAddPatient(false)} aria-label="Close add patient dialog">
                <X size={16} />
              </button>
            </div>
            <input
              className="input"
              required
              placeholder="Patient name"
              value={newPatient.name}
              onChange={(e) => setNewPatient({ ...newPatient, name: e.target.value })}
            />
            <div style={styles.formRow}>
              <input
                className="input"
                type="number"
                min="1"
                max="120"
                placeholder="Age"
                value={newPatient.age}
                onChange={(e) => setNewPatient({ ...newPatient, age: e.target.value })}
              />
              <select
                className="input"
                value={newPatient.gender}
                onChange={(e) => setNewPatient({ ...newPatient, gender: e.target.value })}
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
            <textarea
              className="input"
              rows={3}
              placeholder="Notes"
              value={newPatient.notes}
              onChange={(e) => setNewPatient({ ...newPatient, notes: e.target.value })}
              style={{ resize: 'vertical' }}
            />
            <button className="btn btn-primary" type="submit" style={{ justifyContent: 'center' }}>
              Add Patient
            </button>
          </form>
        </div>
      )}
    </header>
  );
};

const styles: Record<string, React.CSSProperties> = {
  header: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    height: 'var(--nav-height)',
    background: 'rgba(250, 248, 245, 0.92)',
    borderBottom: '1px solid var(--soft-beige)',
    backdropFilter: 'blur(18px)',
    WebkitBackdropFilter: 'blur(18px)',
    zIndex: 1000,
    display: 'flex',
    alignItems: 'center',
    boxShadow: 'var(--shadow-sm)',
  },
  navContainer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    width: '100%',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    textDecoration: 'none',
  },
  logoIcon: {
    width: '32px',
    height: '32px',
    borderRadius: 'var(--radius-sm)',
    background: 'var(--soft-beige)',
    border: '1px solid #d8c7bb',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoText: {
    fontSize: '1.25rem',
    fontWeight: 800,
    letterSpacing: '-0.02em',
    color: 'var(--deep-black)',
  },
  navLinks: {
    display: 'flex',
    gap: '8px',
  },
  link: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    color: 'var(--espresso)',
    textDecoration: 'none',
    fontWeight: 500,
    fontSize: '0.9rem',
    padding: '6px 12px',
    borderRadius: 'var(--radius-sm)',
    transition: 'all var(--transition-fast)',
  },
  activeLink: {
    color: 'var(--deep-black)',
    background: 'var(--soft-beige)',
  },
  patientSelector: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    background: '#fff',
    border: '1px solid var(--soft-beige)',
    borderRadius: 'var(--radius-md)',
    padding: '4px 12px',
    maxWidth: '220px',
  },
  select: {
    background: 'transparent',
    border: 'none',
    color: 'var(--deep-black)',
    fontFamily: 'inherit',
    fontSize: '0.85rem',
    fontWeight: 500,
    outline: 'none',
    cursor: 'pointer',
    width: '100%',
  },
  option: {
    background: 'var(--warm-white)',
    color: 'var(--deep-black)',
  },
  modalBackdrop: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(17, 17, 17, 0.22)',
    display: 'flex',
    justifyContent: 'flex-end',
    alignItems: 'flex-start',
    padding: '76px 186px 24px 24px',
    zIndex: 1001,
  },
  modal: {
    width: '360px',
    background: '#fff',
    border: '1px solid var(--soft-beige)',
    borderRadius: 8,
    boxShadow: 'var(--shadow-lg)',
    padding: 18,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  modalHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingBottom: 8,
    borderBottom: '1px solid var(--soft-beige)',
  },
  iconButton: {
    width: 28,
    height: 28,
    border: '1px solid var(--soft-beige)',
    background: 'var(--warm-white)',
    borderRadius: 8,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
  },
  formRow: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 10,
  },
};
