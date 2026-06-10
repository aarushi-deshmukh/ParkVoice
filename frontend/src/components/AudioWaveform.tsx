import React, { useRef, useState, useEffect } from 'react';
import { Play, Pause, Volume2 } from 'lucide-react';

interface AudioWaveformProps {
  src: string; // URL to the audio file
}

export const AudioWaveform: React.FC<AudioWaveformProps> = ({ src }) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [waveformData, setWaveformData] = useState<number[]>([]);

  // Generate a random but deterministic waveform shape based on the source url length
  useEffect(() => {
    const data: number[] = [];
    const points = 60;
    const seed = src.length || 42;
    let current = 0.4;
    for (let i = 0; i < points; i++) {
      const angle = (i / points) * Math.PI * 4;
      const noise = Math.sin(angle * 1.5 + seed) * 0.2 + Math.cos(angle * 3) * 0.15;
      let val = Math.max(0.1, 0.4 + noise);
      // Fade edges
      if (i < 5) val *= i / 5;
      if (i > points - 6) val *= (points - 1 - i) / 5;
      data.push(val);
    }
    setWaveformData(data);
  }, [src]);

  // Audio lifecycle
  useEffect(() => {
    const audio = new Audio(src);
    audioRef.current = audio;

    const onTimeUpdate = () => setCurrentTime(audio.currentTime);
    const onLoadedMetadata = () => setDuration(audio.duration || 0);
    const onEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };

    audio.addEventListener('timeupdate', onTimeUpdate);
    audio.addEventListener('loadedmetadata', onLoadedMetadata);
    audio.addEventListener('ended', onEnded);

    return () => {
      audio.pause();
      audio.removeEventListener('timeupdate', onTimeUpdate);
      audio.removeEventListener('loadedmetadata', onLoadedMetadata);
      audio.removeEventListener('ended', onEnded);
    };
  }, [src]);

  // Render waveform
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const progress = duration > 0 ? currentTime / duration : 0;
    const barWidth = 4;
    const gap = 2;
    const nBars = waveformData.length;

    for (let i = 0; i < nBars; i++) {
      const x = i * (barWidth + gap);
      const barHeight = waveformData[i] * height;
      const y = (height - barHeight) / 2;

      // Color based on active vs inactive bar
      const isPlayed = (i / nBars) <= progress;
      if (isPlayed) {
        ctx.fillStyle = '#6B4F3A';
      } else {
        ctx.fillStyle = 'rgba(232, 244, 248, 0.15)'; // Muted white
      }

      // Rounded rects for bars
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, 2);
      ctx.fill();
    }
  }, [currentTime, duration, waveformData]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().catch(err => console.error("Playback failed", err));
      setIsPlaying(true);
    }
  };

  const handleScrub = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const audio = audioRef.current;
    if (!canvas || !audio || duration === 0) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const pct = clickX / rect.width;
    const newTime = pct * duration;
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
  };

  return (
    <div style={styles.container}>
      <button onClick={togglePlay} style={styles.playBtn} aria-label={isPlaying ? "Pause" : "Play"}>
        {isPlaying ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
      </button>

      <div style={styles.waveformContainer}>
        <canvas
          ref={canvasRef}
          width={360}
          height={48}
          onClick={handleScrub}
          style={styles.canvas}
        />
      </div>

      <div style={styles.info}>
        <span style={styles.time}>{formatTime(currentTime)} / {formatTime(duration)}</span>
        <Volume2 size={14} style={{ color: 'var(--text-muted)' }} />
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    background: 'rgba(11, 26, 46, 0.4)',
    border: '1px solid var(--glass-border)',
    borderRadius: 'var(--radius-md)',
    padding: '8px 16px',
    width: '100%',
    backdropFilter: 'blur(8px)',
  },
  playBtn: {
    width: '36px',
    height: '36px',
    borderRadius: '50%',
    border: 'none',
    background: 'linear-gradient(135deg, var(--teal-500), var(--blue-500))',
    color: '#fff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    boxShadow: '0 2px 8px rgba(0, 212, 170, 0.3)',
    transition: 'transform 0.15s ease',
  },
  waveformContainer: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
  },
  canvas: {
    cursor: 'pointer',
    width: '100%',
    display: 'block',
  },
  info: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  time: {
    fontSize: '0.78rem',
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-secondary)',
    whiteSpace: 'nowrap',
  },
};
