import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useStore } from '../store/useStore';
import { api } from '../api/client';
import { AudioWaveform } from '../components/AudioWaveform';
import { Mic, Square, RotateCcw, Upload, AlertCircle, Sparkles } from 'lucide-react';

export const Record: React.FC = () => {
  const navigate = useNavigate();
  const { patients, selectedPatientId, setSelectedPatientId, fetchPatients } = useStore();

  const [isRecording, setIsRecording] = useState(false);
  const [recordTime, setRecordTime] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  // Refs for recording logic
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);

  // Refs for Web Audio visualizer
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    fetchPatients();
    return () => {
      stopTimer();
      stopVisualizer();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [fetchPatients]);

  // Visualizer loop
  const startVisualizer = (stream: MediaStream) => {
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const draw = () => {
        if (!analyserRef.current) return;
        animationFrameRef.current = requestAnimationFrame(draw);

        analyserRef.current.getByteFrequencyData(dataArray);

        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);

        const barWidth = (width / bufferLength) * 1.5;
        let barHeight;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          barHeight = (dataArray[i] / 255) * height;

          // Brown-spectrum recording bars
          const g = ctx.createLinearGradient(0, height, 0, height - barHeight);
          g.addColorStop(0, 'var(--blue-500)');
          g.addColorStop(1, 'var(--teal-500)');
          ctx.fillStyle = g;

          // Mirror visualizer
          ctx.beginPath();
          ctx.roundRect(x, (height - barHeight) / 2, barWidth - 1, barHeight, 2);
          ctx.fill();

          x += barWidth;
        }
      };

      draw();
    } catch (e) {
      console.error("Failed to start audio visualizer", e);
    }
  };

  const stopVisualizer = () => {
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    if (audioContextRef.current) audioContextRef.current.close().catch(() => {});
    audioContextRef.current = null;
    analyserRef.current = null;
  };

  // Timer functions
  const startTimer = () => {
    setRecordTime(0);
    timerRef.current = window.setInterval(() => {
      setRecordTime(prev => prev + 1);
    }, 1000);
  };

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  // MediaRecorder triggers
  const startRecording = async () => {
    setError(null);
    setRecordedBlob(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    audioChunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } });
      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setRecordedBlob(audioBlob);
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);

        // Stop stream tracks
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      startTimer();
      startVisualizer(stream);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Could not access microphone. Ensure microphone permissions are enabled.");
    }
  };

  const stopRecording = () => {
    if (!mediaRecorderRef.current || !isRecording) return;
    mediaRecorderRef.current.stop();
    setIsRecording(false);
    stopTimer();
    stopVisualizer();
  };

  const resetRecording = () => {
    setRecordedBlob(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    setRecordTime(0);
    setError(null);
  };

  const handleUpload = async () => {
    if (!recordedBlob) return;
    setIsUploading(true);
    setError(null);

    try {
      const filename = `recording_${new Date().getTime()}.wav`;
      const file = new File([recordedBlob], filename, { type: 'audio/wav' });
      const result = await api.uploadAudio(file, selectedPatientId || undefined);
      setIsUploading(false);
      navigate(`/analysis/${result.id}`);
    } catch (err: any) {
      setIsUploading(false);
      setError(err.response?.data?.detail || err.message || 'Upload failed. Please try again.');
    }
  };

  const formatTime = (time: number) => {
    const min = Math.floor(time / 60);
    const sec = time % 60;
    return `${min}:${sec < 10 ? '0' : ''}${sec}`;
  };

  return (
    <div className="page-layout fade-in">
      <section className="overview-hero page-hero">
        <div>
          <span className="eyebrow">Acoustic Screening Support</span>
          <h1>In-Browser Voice Recorder</h1>
          <p>Record voice samples in real time using the same screening workflow and visual language as the overview dashboard.</p>
        </div>
      </section>

      <div style={styles.grid}>
        {/* Recording Panel */}
        <div className="glass-card" style={styles.recordPanel}>
          <div style={styles.patientRow}>
            <label style={styles.patientLabel}>Patient Context:</label>
            <select
              value={selectedPatientId || ''}
              onChange={(e) => setSelectedPatientId(e.target.value || null)}
              className="input"
              style={{ maxWidth: '280px', height: '40px', padding: '8px 12px' }}
            >
              <option value="">Anonymized Evaluation</option>
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} (Age {p.age || 'N/A'})
                </option>
              ))}
            </select>
          </div>

          <div style={styles.visualizerContainer}>
            {isRecording ? (
              <canvas ref={canvasRef} width={500} height={120} style={styles.canvas} />
            ) : (
              <div style={styles.visualizerFallback}>
                {audioUrl ? (
                  <div style={{ width: '100%', maxWidth: '450px' }}>
                    <p style={styles.playbackHeading}>Voice Clip Playback</p>
                    <AudioWaveform src={audioUrl} />
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', color: 'var(--text-muted)' }}>
                    <Mic size={32} />
                    <span>Microphone Idle</span>
                  </div>
                )}
              </div>
            )}
          </div>

          <div style={styles.timer}>
            <div style={isRecording ? styles.pulseDot : styles.dot} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', fontWeight: 600 }}>
              {formatTime(recordTime)}
            </span>
          </div>

          {/* Action Buttons */}
          <div style={styles.actions}>
            {!isRecording && !audioUrl && (
              <button onClick={startRecording} className="btn btn-primary btn-lg" style={styles.bigBtn}>
                <Mic size={20} />
                <span>Start Recording</span>
              </button>
            )}

            {isRecording && (
              <button onClick={stopRecording} className="btn btn-danger btn-lg" style={styles.bigBtn}>
                <Square size={18} fill="currentColor" />
                <span>Stop Recording</span>
              </button>
            )}

            {!isRecording && audioUrl && (
              <div style={styles.postRecordActions}>
                <button onClick={resetRecording} className="btn btn-ghost" style={{ flex: 1 }}>
                  <RotateCcw size={16} />
                  <span>Retry</span>
                </button>
                <button onClick={handleUpload} disabled={isUploading} className="btn btn-primary" style={{ flex: 2 }}>
                  {isUploading ? (
                    <>
                      <div className="spinner" />
                      <span>Uploading...</span>
                    </>
                  ) : (
                    <>
                      <Upload size={16} />
                      <span>Upload & Analyze</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </div>

          {error && (
            <div style={styles.errorBox}>
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Vocal Phonation Protocol Card */}
        <div className="glass-card" style={styles.protocolCard}>
          <h3 style={styles.protocolTitle}>
            <Sparkles size={18} style={{ color: 'var(--teal-400)' }} />
            <span>Clinical Phonation Protocol</span>
          </h3>
          <div style={styles.step}>
            <div style={styles.stepNum}>1</div>
            <div>
              <h4 style={styles.stepTitle}>Positioning</h4>
              <p style={styles.stepDesc}>Seat the subject upright. Place microphone exactly 15cm from mouth, angled at 45 degrees to prevent breath pops.</p>
            </div>
          </div>
          <div style={styles.step}>
            <div style={styles.stepNum}>2</div>
            <div>
              <h4 style={styles.stepTitle}>Inhalation</h4>
              <p style={styles.stepDesc}>Instruct the subject to take a deep, natural breath filling their chest fully.</p>
            </div>
          </div>
          <div style={styles.step}>
            <div style={styles.stepNum}>3</div>
            <div>
              <h4 style={styles.stepTitle}>Phonation</h4>
              <p style={styles.stepDesc}>Ask the patient to steadily produce the vowel <strong>"ah"</strong> (as in "father") at a comfortable pitch and volume for at least 8 seconds.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  header: {
    marginBottom: '20px',
    textAlign: 'center',
  },
  title: {
    fontSize: '2.2rem',
    fontWeight: 900,
    background: 'var(--gradient-teal)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    marginBottom: '8px',
  },
  subtitle: {
    color: 'var(--text-secondary)',
    maxWidth: '700px',
    margin: '0 auto',
    fontSize: '1rem',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '1.15fr 0.85fr',
    gap: '20px',
  },
  recordPanel: {
    padding: '22px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '18px',
  },
  patientRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    width: '100%',
    justifyContent: 'center',
  },
  patientLabel: {
    fontSize: '0.85rem',
    fontWeight: 600,
    color: 'var(--text-secondary)',
  },
  visualizerContainer: {
    width: '100%',
    height: '150px',
    background: 'var(--soft-beige)',
    borderRadius: '8px',
    border: '1px solid var(--glass-border)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    position: 'relative',
    padding: '16px',
  },
  canvas: {
    width: '100%',
    height: '100%',
    display: 'block',
  },
  visualizerFallback: {
    display: 'flex',
    width: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  playbackHeading: {
    fontSize: '0.8rem',
    fontWeight: 600,
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    marginBottom: '8px',
  },
  timer: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    background: 'var(--warm-white)',
    padding: '6px 16px',
    borderRadius: 'var(--radius-full)',
    border: '1px solid var(--glass-border)',
  },
  dot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    background: 'var(--text-muted)',
  },
  pulseDot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    background: 'var(--red-500)',
    animation: 'pulse-glow 1s infinite',
  },
  actions: {
    width: '100%',
    maxWidth: '400px',
    display: 'flex',
    justifyContent: 'center',
  },
  bigBtn: {
    width: '100%',
    display: 'flex',
    justifyContent: 'center',
    padding: '14px',
    fontSize: '1rem',
  },
  postRecordActions: {
    display: 'flex',
    gap: '12px',
    width: '100%',
  },
  errorBox: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    color: 'var(--red-400)',
    background: 'rgba(58, 42, 32, 0.08)',
    border: '1px solid rgba(58, 42, 32, 0.2)',
    borderRadius: 'var(--radius-md)',
    padding: '12px 16px',
    fontSize: '0.85rem',
    width: '100%',
  },
  protocolCard: {
    padding: '22px',
    display: 'flex',
    flexDirection: 'column',
    gap: '18px',
  },
  protocolTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '1.1rem',
    color: 'var(--text-primary)',
    marginBottom: '8px',
    borderBottom: '1px solid var(--glass-border)',
    paddingBottom: '12px',
  },
  step: {
    display: 'flex',
    gap: '16px',
    alignItems: 'flex-start',
  },
  stepNum: {
    width: '24px',
    height: '24px',
    borderRadius: '50%',
    background: 'rgba(107, 79, 58, 0.1)',
    border: '1px solid rgba(107, 79, 58, 0.3)',
    color: 'var(--teal-400)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 700,
    fontSize: '0.85rem',
    flexShrink: 0,
  },
  stepTitle: {
    fontSize: '0.92rem',
    fontWeight: 700,
    color: 'var(--text-primary)',
    marginBottom: '2px',
  },
  stepDesc: {
    fontSize: '0.82rem',
    color: 'var(--text-secondary)',
    lineHeight: 1.4,
  },
};
