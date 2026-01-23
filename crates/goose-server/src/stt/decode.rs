//! Audio decoding from various formats to PCM samples
//!
//! Supports WAV and WebM (with Opus codec via libopus).

use super::{Error, Result};
use std::io::Cursor;
use std::sync::LazyLock;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CodecRegistry, DecoderOptions};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::{Hint, Probe};

/// Custom codec registry with libopus support
static CODEC_REGISTRY: LazyLock<CodecRegistry> = LazyLock::new(|| {
    let mut registry = CodecRegistry::new();
    // Register built-in PCM codec
    registry.register_all::<symphonia::default::codecs::PcmDecoder>();
    // Register libopus adapter for Opus decoding
    registry.register_all::<symphonia_adapter_libopus::OpusDecoder>();
    registry
});

/// Custom probe with supported formats
static PROBE: LazyLock<Probe> = LazyLock::new(|| {
    let mut probe = Probe::default();
    // Register WAV format
    probe.register_all::<symphonia::default::formats::WavReader>();
    // Register MKV/WebM format
    probe.register_all::<symphonia::default::formats::MkvReader>();
    probe
});

/// Decoded audio data
pub struct DecodedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Decode audio bytes to PCM samples
///
/// Supported formats: WAV, WebM (with Opus codec)
pub fn decode_audio(bytes: &[u8], mime_type: &str) -> Result<DecodedAudio> {
    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    // Provide format hint based on MIME type (only wav and webm supported)
    let mut hint = Hint::new();
    match mime_type {
        "audio/webm" => hint.with_extension("webm"),
        "audio/wav" | "audio/wave" => hint.with_extension("wav"),
        _ => return Err(Error::UnsupportedFormat(format!(
            "{}. Only audio/wav and audio/webm are supported.",
            mime_type
        ))),
    };

    // Probe the format using our custom probe
    let probed = PROBE
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| Error::AudioDecode(format!("Failed to probe format: {}", e)))?;

    let mut format = probed.format;

    // Find the first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| Error::AudioDecode("No audio track found".to_string()))?;

    let codec_params = track.codec_params.clone();
    let track_id = track.id;

    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| Error::AudioDecode("Unknown sample rate".to_string()))?;

    let channels = codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(1);

    // Create decoder using our custom codec registry (with libopus)
    let mut decoder = CODEC_REGISTRY
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| Error::AudioDecode(format!("Failed to create decoder: {}", e)))?;

    let mut all_samples: Vec<f32> = Vec::new();

    // Decode all packets
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                return Err(Error::AudioDecode(format!("Failed to read packet: {}", e)));
            }
        };

        // Skip packets from other tracks
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => {
                return Err(Error::AudioDecode(format!("Failed to decode packet: {}", e)));
            }
        };

        // Convert to f32 samples
        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;

        let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
        sample_buf.copy_interleaved_ref(decoded);

        all_samples.extend_from_slice(sample_buf.samples());
    }

    Ok(DecodedAudio {
        samples: all_samples,
        sample_rate,
        channels,
    })
}

/// Convert interleaved multi-channel audio to mono
pub fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }

    samples
        .chunks(channels as usize)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}
