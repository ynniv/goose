//! Core transcriber implementation

use super::decode::{decode_audio, to_mono};
use super::{Error, Result};
use parakeet_rs::Nemotron;
use std::path::Path;

const CHUNK_SAMPLES: usize = 8960; // 560ms at 16kHz

/// Configuration for the transcriber
#[derive(Debug, Clone)]
pub struct TranscriberConfig {
    /// Whether to apply punctuation processing
    pub punctuation: bool,
}

impl Default for TranscriberConfig {
    fn default() -> Self {
        Self { punctuation: true }
    }
}

/// Speech-to-text transcriber using Nemotron model
pub struct Transcriber {
    nemotron: Nemotron,
    config: TranscriberConfig,
}

impl Transcriber {
    /// Create a new transcriber with the model at the given path
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        Self::with_config(model_dir, TranscriberConfig::default())
    }

    /// Create a new transcriber with custom configuration
    pub fn with_config<P: AsRef<Path>>(model_dir: P, config: TranscriberConfig) -> Result<Self> {
        let nemotron = Nemotron::from_pretrained(model_dir.as_ref(), None)
            .map_err(|e| Error::ModelLoad(e.to_string()))?;

        Ok(Self { nemotron, config })
    }

    /// Transcribe audio bytes with the given MIME type
    ///
    /// Audio must be 16kHz sample rate (resampling should be done client-side).
    ///
    /// Supported MIME types:
    /// - audio/webm (with opus codec)
    /// - audio/wav, audio/wave
    pub fn transcribe(&mut self, audio_bytes: &[u8], mime_type: &str) -> Result<String> {
        // Decode audio to PCM
        let decoded = decode_audio(audio_bytes, mime_type)?;

        // Verify sample rate is 16kHz (allow small tolerance for rounding)
        if decoded.sample_rate < 15900 || decoded.sample_rate > 16100 {
            return Err(Error::AudioDecode(format!(
                "Audio must be 16kHz, got {}Hz. Please resample client-side.",
                decoded.sample_rate
            )));
        }

        // Convert to mono if needed
        let samples = to_mono(&decoded.samples, decoded.channels);

        // Transcribe using the model
        self.transcribe_samples(&samples)
    }

    /// Transcribe pre-processed samples (16kHz mono f32)
    pub fn transcribe_samples(&mut self, samples: &[f32]) -> Result<String> {
        // Reset state for new transcription
        self.nemotron.reset();

        // Process in chunks
        for chunk in samples.chunks(CHUNK_SAMPLES) {
            let mut chunk_vec = chunk.to_vec();

            // Pad last chunk if needed
            if chunk_vec.len() < CHUNK_SAMPLES {
                chunk_vec.resize(CHUNK_SAMPLES, 0.0);
            }

            self.nemotron
                .transcribe_chunk(&chunk_vec)
                .map_err(|e| Error::Transcribe(e.to_string()))?;
        }

        let mut transcript = self.nemotron.get_transcript();

        if self.config.punctuation {
            transcript = super::punctuation::process(&transcript);
        }

        Ok(transcript.trim().to_string())
    }

    /// Get the underlying Nemotron model for streaming/advanced use
    #[allow(dead_code)]
    pub fn nemotron(&mut self) -> &mut Nemotron {
        &mut self.nemotron
    }

    /// Reset the transcription state
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.nemotron.reset();
    }
}
