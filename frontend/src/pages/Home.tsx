import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, FileAudio, Upload } from 'lucide-react';
import { api } from '../api/client';
import { useStore } from '../store/useStore';

export const Home: React.FC = () => {
  const navigate = useNavigate();
  const { selectedPatientId } = useStore();
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const onFile = (candidate?: File) => {
    if (!candidate) return;
    const ext = candidate.name.split('.').pop()?.toLowerCase();
    if (!ext || !['wav', 'mp3', 'm4a', 'ogg', 'flac'].includes(ext)) {
      setError('Upload WAV, MP3, M4A, OGG, or FLAC audio.');
      setFile(null);
      return;
    }
    setError(null);
    setFile(candidate);
  };

  const submit = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const result = await api.uploadAudio(file, selectedPatientId ?? undefined);
      navigate(`/analysis/${result.id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail ?? err.message ?? 'Upload failed.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="page-layout overview-page">
      <section className="overview-hero">
        <div>
          <span className="eyebrow">Acoustic Screening Support</span>
          <h1>Screen a Voice Recording</h1>
          <p>This system supports research screening workflows only. It does not diagnose Parkinson's disease.</p>
        </div>
      </section>

      <section className="panel">
        <h2>Upload Audio</h2>
        <label className="upload-surface">
          <Upload size={30} />
          <strong>{file ? file.name : 'Choose sustained vowel or short speech audio'}</strong>
          <span>WAV, MP3, M4A, OGG, or FLAC up to 50 MB</span>
          <input type="file" accept=".wav,.mp3,.m4a,.ogg,.flac" onChange={(event) => onFile(event.target.files?.[0])} />
        </label>
        {file && (
          <div className="benchmark-row">
            <FileAudio size={16} />
            <span>{(file.size / (1024 * 1024)).toFixed(2)} MB ready for screening</span>
            <button className="btn btn-primary" onClick={submit} disabled={uploading}>{uploading ? 'Uploading...' : 'Run Assessment'}</button>
          </div>
        )}
        {error && <div className="disclaimer"><AlertCircle size={16} /> {error}</div>}
      </section>
    </div>
  );
};
