import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { Overview } from './pages/Overview';
import { Home } from './pages/Home';
import { Record } from './pages/Record';
import { AnalysisPage } from './pages/Analysis';
import { Report } from './pages/Report';
import { Explainability } from './pages/Explainability';
import { History } from './pages/History';
import { Biomarkers } from './pages/Biomarkers';
import { Benchmarks } from './pages/Benchmarks';

function App() {
  return (
    <Router>
      <div style={styles.appContainer}>
        {/* Navigation Bar */}
        <Navbar />

        {/* Page Content area with padding to clear navigation bar */}
        <main style={styles.mainContent}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/home" element={<Home />} />
            <Route path="/record" element={<Record />} />
            <Route path="/analysis/:id" element={<AnalysisPage />} />
            <Route path="/report/:id" element={<Report />} />
            <Route path="/explain/:id" element={<Explainability />} />
            <Route path="/history" element={<History />} />
            <Route path="/biomarkers" element={<Biomarkers />} />
            <Route path="/benchmarks" element={<Benchmarks />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

const styles: Record<string, React.CSSProperties> = {
  appContainer: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    background: 'var(--navy-950)',
    backgroundColor: 'var(--warm-white)',
  },
  mainContent: {
    flex: 1,
    paddingTop: 'var(--nav-height)', // Make sure pages don't go under fixed header
  },
};

export default App;
