import { useState, useEffect, useCallback } from 'react';
import { Download, Trash2, CheckCircle, AlertCircle, Loader2, Cpu, Brain } from 'lucide-react';
import { getApiUrl } from '../../../config';

interface ModelFile {
  name: string;
  exists: boolean;
  size: number;
}

interface ModelStatus {
  installed: boolean;
  path: string;
  total_size: number;
  files: ModelFile[];
}

interface RuntimeStatus {
  installed: boolean;
  version: string;
  path: string;
  size: number;
  platform_supported: boolean;
}

interface DownloadProgress {
  file?: string;
  file_index?: number;
  total_files?: number;
  status: 'starting' | 'downloading' | 'complete';
  bytes_downloaded: number;
  total_bytes: number;
}

export const LocalModelManager = () => {
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloadingRuntime, setDownloadingRuntime] = useState(false);
  const [downloadingModel, setDownloadingModel] = useState(false);
  const [deleting, setDeleting] = useState<'runtime' | 'model' | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const fetchStatus = useCallback(async () => {
    try {
      const secretKey = await window.electron.getSecretKey();
      const headers = { 'X-Secret-Key': secretKey };

      const [runtimeRes, modelRes] = await Promise.all([
        fetch(getApiUrl('/audio/runtime/status'), { headers }),
        fetch(getApiUrl('/audio/model/status'), { headers }),
      ]);

      if (runtimeRes.ok) {
        setRuntimeStatus(await runtimeRes.json());
      }
      if (modelRes.ok) {
        setModelStatus(await modelRes.json());
      }
    } catch (err) {
      console.error('Failed to fetch status:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const handleSSEDownload = async (endpoint: string, setDownloading: (v: boolean) => void) => {
    setDownloading(true);
    setError(null);
    setProgress(null);

    try {
      const secretKey = await window.electron.getSecretKey();

      const response = await fetch(getApiUrl(endpoint), {
        method: 'POST',
        headers: { 'X-Secret-Key': secretKey },
      });

      if (!response.ok) {
        throw new Error(`Download failed: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new window.TextDecoder();

      if (!reader) {
        throw new Error('No response body');
      }

      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: error')) {
            // Next data line will have the error
            continue;
          }
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            try {
              const parsed = JSON.parse(data);
              setProgress(parsed);
            } catch {
              if (data.includes('error') || data.includes('Failed')) {
                setError(data);
                setDownloading(false);
                return;
              }
            }
          }
        }
      }

      setDownloading(false);
      setProgress(null);
      fetchStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed');
      setDownloading(false);
    }
  };

  const handleDelete = async (type: 'runtime' | 'model') => {
    const message =
      type === 'runtime'
        ? 'Are you sure you want to delete ONNX Runtime? The model will not work without it.'
        : 'Are you sure you want to delete the model?';

    if (!window.confirm(message)) {
      return;
    }

    setDeleting(type);
    setError(null);

    try {
      const endpoint = type === 'runtime' ? '/audio/runtime/clean' : '/audio/model/clean';
      const response = await fetch(getApiUrl(endpoint), {
        method: 'POST',
        headers: { 'X-Secret-Key': await window.electron.getSecretKey() },
      });

      if (response.ok) {
        fetchStatus();
      } else {
        setError(`Failed to delete ${type}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed');
    } finally {
      setDeleting(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-text-muted text-xs">
        <Loader2 className="w-3 h-3 animate-spin" />
        Checking status...
      </div>
    );
  }

  const downloading = downloadingRuntime || downloadingModel;
  const modelProgress =
    progress && progress.total_files
      ? ((progress.file_index! + progress.bytes_downloaded / Math.max(progress.total_bytes, 1)) /
          progress.total_files) *
        100
      : progress
        ? (progress.bytes_downloaded / Math.max(progress.total_bytes, 1)) * 100
        : 0;

  return (
    <div className="mt-3 space-y-4">
      {/* Error message */}
      {error && <div className="text-xs text-red-500 bg-red-500/10 p-2 rounded">{error}</div>}

      {/* ONNX Runtime Section */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-text-muted" />
          <span className="text-xs font-medium text-text-default">ONNX Runtime</span>
          {runtimeStatus?.installed ? (
            <CheckCircle className="w-3 h-3 text-green-500" />
          ) : (
            <AlertCircle className="w-3 h-3 text-yellow-500" />
          )}
        </div>

        {runtimeStatus?.installed ? (
          <div className="flex items-center gap-2 ml-6">
            <span className="text-xs text-text-muted">
              v{runtimeStatus.version} ({formatBytes(runtimeStatus.size)})
            </span>
            {!downloading && (
              <button
                onClick={() => handleDelete('runtime')}
                disabled={deleting === 'runtime'}
                className="text-xs text-red-500 hover:text-red-400 disabled:opacity-50"
              >
                {deleting === 'runtime' ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <Trash2 className="w-3 h-3" />
                )}
              </button>
            )}
          </div>
        ) : (
          <div className="ml-6">
            {downloadingRuntime && progress ? (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-text-muted">
                  <span>Downloading...</span>
                  <span>
                    {formatBytes(progress.bytes_downloaded)} / {formatBytes(progress.total_bytes)}
                  </span>
                </div>
                <div className="w-full bg-background-default rounded-full h-1.5">
                  <div
                    className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${modelProgress}%` }}
                  />
                </div>
              </div>
            ) : runtimeStatus?.platform_supported ? (
              <button
                onClick={() => handleSSEDownload('/audio/runtime/download', setDownloadingRuntime)}
                disabled={downloading}
                className="flex items-center gap-1.5 px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                <Download className="w-3 h-3" />
                Download (~80 MB)
              </button>
            ) : (
              <span className="text-xs text-red-500">Platform not supported</span>
            )}
          </div>
        )}
      </div>

      {/* Nemotron Model Section */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-text-muted" />
          <span className="text-xs font-medium text-text-default">Nemotron Model</span>
          {modelStatus?.installed ? (
            <CheckCircle className="w-3 h-3 text-green-500" />
          ) : (
            <AlertCircle className="w-3 h-3 text-yellow-500" />
          )}
        </div>

        {modelStatus?.installed ? (
          <div className="flex items-center gap-2 ml-6">
            <span className="text-xs text-text-muted">
              Installed ({formatBytes(modelStatus.total_size)})
            </span>
            {!downloading && (
              <button
                onClick={() => handleDelete('model')}
                disabled={deleting === 'model'}
                className="text-xs text-red-500 hover:text-red-400 disabled:opacity-50"
              >
                {deleting === 'model' ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <Trash2 className="w-3 h-3" />
                )}
              </button>
            )}
          </div>
        ) : (
          <div className="ml-6">
            {downloadingModel && progress ? (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-text-muted">
                  <span>
                    {progress.file} ({(progress.file_index ?? 0) + 1}/{progress.total_files})
                  </span>
                  <span>
                    {formatBytes(progress.bytes_downloaded)} / {formatBytes(progress.total_bytes)}
                  </span>
                </div>
                <div className="w-full bg-background-default rounded-full h-1.5">
                  <div
                    className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${modelProgress}%` }}
                  />
                </div>
              </div>
            ) : (
              <button
                onClick={() => handleSSEDownload('/audio/model/download', setDownloadingModel)}
                disabled={downloading || !runtimeStatus?.installed}
                className="flex items-center gap-1.5 px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
                title={!runtimeStatus?.installed ? 'Install ONNX Runtime first' : undefined}
              >
                <Download className="w-3 h-3" />
                Download (~2.4 GB)
              </button>
            )}
            {!runtimeStatus?.installed && (
              <span className="text-xs text-text-muted ml-2">(requires ONNX Runtime)</span>
            )}
          </div>
        )}
      </div>

      {/* Ready indicator */}
      {runtimeStatus?.installed && modelStatus?.installed && (
        <div className="flex items-center gap-2 p-2 bg-green-500/10 rounded">
          <CheckCircle className="w-4 h-4 text-green-500" />
          <span className="text-xs text-green-600">Ready for local transcription</span>
        </div>
      )}
    </div>
  );
};
