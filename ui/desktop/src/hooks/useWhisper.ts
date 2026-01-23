import { useState, useRef, useCallback, useEffect } from 'react';
import { useConfig } from '../components/ConfigContext';
import { getApiUrl } from '../config';
import { useDictationSettings } from './useDictationSettings';
import { safeJsonParse } from '../utils/conversionUtils';

interface UseWhisperOptions {
  onTranscription?: (text: string) => void;
  onError?: (error: Error) => void;
  onSizeWarning?: (sizeInMB: number) => void;
}

// Constants
const MAX_AUDIO_SIZE_MB = 25;
const TARGET_SAMPLE_RATE = 16000; // Nemotron model requires 16kHz

/**
 * Convert an audio blob to 16kHz mono WAV format using Web Audio API.
 * This resamples and converts to mono for the Nemotron STT model.
 */
async function convertToWav(audioBlob: Blob): Promise<Blob> {
  const audioContext = new AudioContext();
  try {
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    // Calculate resampled length
    const resampledLength = Math.ceil(
      (audioBuffer.length * TARGET_SAMPLE_RATE) / audioBuffer.sampleRate
    );

    // Use OfflineAudioContext to resample to 16kHz mono
    const offlineContext = new window.OfflineAudioContext(1, resampledLength, TARGET_SAMPLE_RATE);

    // Create buffer source
    const source = offlineContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineContext.destination);
    source.start(0);

    // Render resampled audio
    const resampledBuffer = await offlineContext.startRendering();
    const monoSamples = resampledBuffer.getChannelData(0);

    // Convert to 16-bit PCM
    const pcmData = new Int16Array(monoSamples.length);
    for (let i = 0; i < monoSamples.length; i++) {
      const s = Math.max(-1, Math.min(1, monoSamples[i]));
      pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }

    // Create WAV file (16kHz mono)
    const wavBuffer = new ArrayBuffer(44 + pcmData.length * 2);
    const view = new DataView(wavBuffer);

    // WAV header
    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + pcmData.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, 1, true); // audio format (PCM)
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, TARGET_SAMPLE_RATE, true);
    view.setUint32(28, TARGET_SAMPLE_RATE * 2, true); // byte rate (16kHz * 1 channel * 2 bytes)
    view.setUint16(32, 2, true); // block align (1 channel * 2 bytes)
    view.setUint16(34, 16, true); // bits per sample
    writeString(36, 'data');
    view.setUint32(40, pcmData.length * 2, true);

    // Write PCM data
    const pcmBytes = new Uint8Array(wavBuffer, 44);
    const pcmView = new DataView(pcmData.buffer);
    for (let i = 0; i < pcmData.length; i++) {
      pcmBytes[i * 2] = pcmView.getUint8(i * 2);
      pcmBytes[i * 2 + 1] = pcmView.getUint8(i * 2 + 1);
    }

    return new Blob([wavBuffer], { type: 'audio/wav' });
  } finally {
    await audioContext.close();
  }
}
const MAX_RECORDING_DURATION_SECONDS = 600; // 10 minutes
const WARNING_SIZE_MB = 20; // Warn at 20MB

export const useWhisper = ({ onTranscription, onError, onSizeWarning }: UseWhisperOptions = {}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [hasOpenAIKey, setHasOpenAIKey] = useState(false);
  const [hasLocalModel, setHasLocalModel] = useState(false);
  const [canUseDictation, setCanUseDictation] = useState(false);
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [estimatedSize, setEstimatedSize] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const recordingStartTimeRef = useRef<number | null>(null);
  const durationIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const currentSizeRef = useRef<number>(0);

  const { getProviders } = useConfig();
  const { settings: dictationSettings, hasElevenLabsKey } = useDictationSettings();

  // Check if OpenAI API key is configured (regardless of current provider)
  useEffect(() => {
    const checkOpenAIKey = async () => {
      try {
        // Get all configured providers
        const providers = await getProviders(false);

        // Find OpenAI provider
        const openAIProvider = providers.find((p) => p.name === 'openai');

        // Check if OpenAI is configured
        if (openAIProvider && openAIProvider.is_configured) {
          setHasOpenAIKey(true);
        } else {
          setHasOpenAIKey(false);
        }
      } catch (error) {
        console.error('Error checking OpenAI configuration:', error);
        setHasOpenAIKey(false);
      }
    };

    checkOpenAIKey();
  }, [getProviders]); // Re-check when providers change

  // Check if local model is available
  useEffect(() => {
    const checkLocalModel = async () => {
      try {
        const response = await fetch(getApiUrl('/audio/config'), {
          headers: {
            'X-Secret-Key': await window.electron.getSecretKey(),
          },
        });
        if (response.ok) {
          const config = await response.json();
          setHasLocalModel(config.local === true);
        }
      } catch (error) {
        console.error('Error checking local model availability:', error);
        setHasLocalModel(false);
      }
    };

    checkLocalModel();
  }, []);

  // Check if dictation can be used based on settings
  useEffect(() => {
    if (!dictationSettings) {
      setCanUseDictation(false);
      return;
    }

    if (!dictationSettings.enabled) {
      setCanUseDictation(false);
      return;
    }

    // Check provider availability
    switch (dictationSettings.provider) {
      case 'openai':
        setCanUseDictation(hasOpenAIKey);
        break;
      case 'elevenlabs':
        setCanUseDictation(hasElevenLabsKey);
        break;
      case 'local':
        setCanUseDictation(hasLocalModel);
        break;
      default:
        setCanUseDictation(false);
    }
  }, [dictationSettings, hasOpenAIKey, hasElevenLabsKey, hasLocalModel]);

  // Define stopRecording before startRecording to avoid circular dependency
  const stopRecording = useCallback(() => {
    setIsRecording(false);

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    // Clear interval
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current);
      durationIntervalRef.current = null;
    }

    // Stop all tracks in the stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    // Close audio context
    if (audioContext && audioContext.state !== 'closed') {
      audioContext.close().catch(console.error);
      setAudioContext(null);
      setAnalyser(null);
    }
  }, [audioContext]);

  // Cleanup effect to prevent memory leaks
  useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (audioContext && audioContext.state !== 'closed') {
        audioContext.close().catch(console.error);
      }
    };
  }, [audioContext]);

  const transcribeAudio = useCallback(
    async (audioBlob: Blob) => {
      if (!dictationSettings) {
        stopRecording();
        onError?.(new Error('Dictation settings not loaded'));
        return;
      }

      setIsTranscribing(true);

      try {
        // Check final size
        const sizeMB = audioBlob.size / (1024 * 1024);
        if (sizeMB > MAX_AUDIO_SIZE_MB) {
          throw new Error(
            `Audio file too large (${sizeMB.toFixed(1)}MB). Maximum size is ${MAX_AUDIO_SIZE_MB}MB.`
          );
        }

        // For local provider, convert to WAV format since symphonia doesn't support opus codec
        let processedBlob = audioBlob;
        if (dictationSettings.provider === 'local' && audioBlob.type.includes('webm')) {
          processedBlob = await convertToWav(audioBlob);
        }

        // Convert blob to base64 for easier transport
        const reader = new FileReader();
        const base64Audio = await new Promise<string>((resolve, reject) => {
          reader.onloadend = () => {
            const base64 = reader.result as string;
            resolve(base64.split(',')[1]); // Remove data:audio/webm;base64, prefix
          };
          reader.onerror = reject;
          reader.readAsDataURL(processedBlob);
        });

        const mimeType = processedBlob.type;
        if (!mimeType) {
          throw new Error('Unable to determine audio format. Please try again.');
        }

        let endpoint = '';
        let headers: Record<string, string> = {
          'Content-Type': 'application/json',
          'X-Secret-Key': await window.electron.getSecretKey(),
        };

        let body: Record<string, string> = {
          audio: base64Audio,
          mime_type: mimeType,
        };

        // Choose endpoint based on provider
        switch (dictationSettings.provider) {
          case 'openai':
            endpoint = '/audio/transcribe';
            break;
          case 'elevenlabs':
            endpoint = '/audio/transcribe/elevenlabs';
            break;
          case 'local':
            endpoint = '/audio/transcribe/local';
            break;
          default:
            throw new Error(`Unsupported provider: ${dictationSettings.provider}`);
        }

        const response = await fetch(getApiUrl(endpoint), {
          method: 'POST',
          headers,
          body: JSON.stringify(body),
        });

        if (!response.ok) {
          if (response.status === 404) {
            throw new Error(
              `Audio transcription endpoint not found. Please implement ${endpoint} endpoint in the Goose backend.`
            );
          } else if (response.status === 401) {
            throw new Error('Invalid API key. Please check your API key is correct.');
          } else if (response.status === 402) {
            throw new Error('API quota exceeded. Please check your account limits.');
          }
          const errorData = await safeJsonParse<{
            error: { message: string };
          }>(response, 'Failed to parse error response').catch(() => ({
            error: { message: 'Transcription failed' },
          }));
          throw new Error(errorData.error?.message || 'Transcription failed');
        }

        const data = await safeJsonParse<{ text: string }>(
          response,
          'Failed to parse transcription response'
        );
        if (data.text) {
          onTranscription?.(data.text);
        }
      } catch (error) {
        console.error('Error transcribing audio:', error);
        stopRecording();
        onError?.(error as Error);
      } finally {
        setIsTranscribing(false);
        setRecordingDuration(0);
        setEstimatedSize(0);
      }
    },
    [onTranscription, onError, dictationSettings, stopRecording]
  );

  const startRecording = useCallback(async () => {
    if (!dictationSettings) {
      stopRecording();
      onError?.(new Error('Dictation settings not loaded'));
      return;
    }

    try {
      // Request microphone permission
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 44100,
        },
      });
      streamRef.current = stream;

      // Verify we have valid audio tracks
      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length === 0) {
        throw new Error('No audio tracks available in the microphone stream');
      }

      // AudioContext creation is disabled to prevent MediaRecorder conflicts
      setAudioContext(null);
      setAnalyser(null);

      // Determine best supported MIME type
      const supportedTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/wav'];

      const mimeType = supportedTypes.find((type) => MediaRecorder.isTypeSupported(type)) || '';

      const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      currentSizeRef.current = 0;
      recordingStartTimeRef.current = Date.now();

      // Start duration and size tracking
      durationIntervalRef.current = setInterval(() => {
        const elapsed = (Date.now() - (recordingStartTimeRef.current || 0)) / 1000;
        setRecordingDuration(elapsed);

        // Estimate size based on typical webm bitrate (~128kbps)
        const estimatedSizeMB = (elapsed * 128 * 1024) / (8 * 1024 * 1024);
        setEstimatedSize(estimatedSizeMB);

        // Check if we're approaching the limit
        if (estimatedSizeMB > WARNING_SIZE_MB) {
          onSizeWarning?.(estimatedSizeMB);
        }

        // Auto-stop if we hit the duration limit
        if (elapsed >= MAX_RECORDING_DURATION_SECONDS) {
          stopRecording();
          onError?.(
            new Error(
              `Maximum recording duration (${MAX_RECORDING_DURATION_SECONDS / 60} minutes) reached.`
            )
          );
        }
      }, 100);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
          currentSizeRef.current += event.data.size;

          // Check actual size
          const actualSizeMB = currentSizeRef.current / (1024 * 1024);
          if (actualSizeMB > MAX_AUDIO_SIZE_MB) {
            stopRecording();
            onError?.(new Error(`Maximum file size (${MAX_AUDIO_SIZE_MB}MB) reached.`));
          }
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType || 'audio/webm' });

        // Check if the blob is empty
        if (audioBlob.size === 0) {
          onError?.(
            new Error(
              'No audio data was recorded. Please check your microphone permissions and try again.'
            )
          );
          return;
        }

        await transcribeAudio(audioBlob);
      };

      // Add error handler for MediaRecorder
      mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        onError?.(new Error('Recording failed: Unknown error'));
      };

      if (!stream.active) {
        throw new Error('Audio stream became inactive before recording could start');
      }

      // Check audio tracks again before starting recording
      if (audioTracks.length === 0) {
        throw new Error('No audio tracks available in the stream');
      }

      const activeAudioTracks = audioTracks.filter((track) => track.readyState === 'live');
      if (activeAudioTracks.length === 0) {
        throw new Error('No live audio tracks available');
      }

      try {
        mediaRecorder.start(100);
        setIsRecording(true);
      } catch (startError) {
        console.error('Error calling mediaRecorder.start():', startError);
        const errorMessage = startError instanceof Error ? startError.message : String(startError);
        throw new Error(`Failed to start recording: ${errorMessage}`);
      }
    } catch (error) {
      console.error('Error starting recording:', error);
      stopRecording();
      onError?.(error as Error);
    }
  }, [onError, onSizeWarning, transcribeAudio, stopRecording, dictationSettings]);

  return {
    isRecording,
    isTranscribing,
    hasOpenAIKey,
    canUseDictation,
    audioContext,
    analyser,
    startRecording,
    stopRecording,
    recordingDuration,
    estimatedSize,
  };
};
