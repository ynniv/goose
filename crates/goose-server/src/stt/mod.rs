//! Speech-to-text module using NVIDIA Nemotron 0.6B via ONNX Runtime.
//!
//! This module provides local, on-device speech-to-text transcription
//! without requiring external API services.

mod decode;
pub mod local;
mod punctuation;
mod transcriber;

pub use transcriber::Transcriber;

/// Error types for speech-to-text
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    #[error("Failed to decode audio: {0}")]
    AudioDecode(String),

    #[error("Failed to transcribe: {0}")]
    Transcribe(String),

    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
