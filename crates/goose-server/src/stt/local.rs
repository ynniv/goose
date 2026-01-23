//! Local speech-to-text using Nemotron model
//!
//! This module handles local transcription using NVIDIA's Nemotron 0.6B model
//! via ONNX Runtime. It manages:
//! - ONNX Runtime dynamic library download and initialization
//! - Nemotron model file download and management
//! - Local transcription endpoint

use super::Transcriber;
use axum::{http::StatusCode, response::Sse, Json};
use futures::stream::Stream;
use reqwest::Client;
use std::convert::Infallible;
use std::time::Duration;
use tokio::sync::OnceCell;

// Lazy-initialized local transcriber
static LOCAL_TRANSCRIBER: OnceCell<std::sync::Mutex<Transcriber>> = OnceCell::const_new();

// Track if ONNX Runtime has been initialized
static ORT_INITIALIZED: OnceCell<bool> = OnceCell::const_new();

/// ONNX Runtime version we target
const ORT_VERSION: &str = "1.23.2";

// ============================================================================
// ONNX Runtime Management
// ============================================================================

/// Get the ONNX Runtime library directory path
fn get_ort_lib_path() -> String {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH") {
        return path;
    }
    goose::config::paths::Paths::in_data_dir("onnxruntime")
        .to_string_lossy()
        .to_string()
}

/// Get the ONNX Runtime library filename for the current platform
fn get_ort_lib_filename() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "libonnxruntime.dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "libonnxruntime.so"
    }
    #[cfg(target_os = "windows")]
    {
        "onnxruntime.dll"
    }
}

/// Get the ONNX Runtime download URL for the current platform
fn get_ort_download_url() -> Option<&'static str> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        Some("https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-arm64-1.23.2.tgz")
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        Some("https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-x86_64-1.23.2.tgz")
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        Some("https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz")
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        Some("https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-aarch64-1.23.2.tgz")
    }
    #[cfg(target_os = "windows")]
    {
        Some("https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-1.23.2.zip")
    }
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        target_os = "windows"
    )))]
    {
        None
    }
}

/// Check if ONNX Runtime library exists
pub fn is_ort_installed() -> bool {
    let lib_path = std::path::Path::new(&get_ort_lib_path()).join(get_ort_lib_filename());
    lib_path.exists()
}

/// Initialize ONNX Runtime from the dynamic library
fn init_ort() -> Result<(), String> {
    if ORT_INITIALIZED.get().is_some() {
        return Ok(());
    }

    let lib_dir = get_ort_lib_path();
    let lib_path = std::path::Path::new(&lib_dir).join(get_ort_lib_filename());

    if !lib_path.exists() {
        return Err(format!(
            "ONNX Runtime library not found at {}. Please download it first.",
            lib_path.display()
        ));
    }

    // Initialize ort with the dynamic library
    let builder = ort::init_from(lib_path.to_string_lossy().as_ref())
        .map_err(|e| format!("Failed to load ONNX Runtime: {}", e))?;

    builder.commit();

    // Mark as initialized (ignore if already set by another thread)
    let _ = ORT_INITIALIZED.set(true);

    tracing::info!("ONNX Runtime initialized from {}", lib_path.display());
    Ok(())
}

/// Get ONNX Runtime status
pub async fn get_ort_status() -> Result<Json<serde_json::Value>, StatusCode> {
    let lib_path = std::path::Path::new(&get_ort_lib_path()).join(get_ort_lib_filename());
    let installed = lib_path.exists();

    let size = if installed {
        std::fs::metadata(&lib_path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    Ok(Json(serde_json::json!({
        "installed": installed,
        "version": ORT_VERSION,
        "path": lib_path.to_string_lossy(),
        "size": size,
        "platform_supported": get_ort_download_url().is_some()
    })))
}

/// Delete ONNX Runtime library
pub async fn delete_ort() -> Result<Json<serde_json::Value>, StatusCode> {
    let lib_dir = get_ort_lib_path();
    let lib_path = std::path::Path::new(&lib_dir).join(get_ort_lib_filename());

    if !lib_path.exists() {
        return Ok(Json(serde_json::json!({
            "success": true,
            "message": "ONNX Runtime was not installed"
        })));
    }

    // Try to delete the entire directory
    match std::fs::remove_dir_all(&lib_dir) {
        Ok(_) => Ok(Json(serde_json::json!({
            "success": true,
            "deleted": lib_dir
        }))),
        Err(e) => Ok(Json(serde_json::json!({
            "success": false,
            "error": format!("Failed to delete: {}", e)
        }))),
    }
}

/// Download ONNX Runtime with SSE progress updates
#[allow(clippy::too_many_lines)]
pub async fn download_ort() -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>>
{
    let stream = async_stream::stream! {
        let download_url = match get_ort_download_url() {
            Some(url) => url,
            None => {
                yield Ok(axum::response::sse::Event::default()
                    .event("error")
                    .data("ONNX Runtime download not supported for this platform"));
                return;
            }
        };

        let lib_dir = get_ort_lib_path();
        let lib_filename = get_ort_lib_filename();

        // Create directory if it doesn't exist
        if let Err(e) = std::fs::create_dir_all(&lib_dir) {
            yield Ok(axum::response::sse::Event::default()
                .event("error")
                .data(format!("Failed to create directory: {}", e)));
            return;
        }

        yield Ok(axum::response::sse::Event::default()
            .event("status")
            .data("Downloading ONNX Runtime..."));

        let client = match Client::builder()
            .timeout(Duration::from_secs(3600))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                yield Ok(axum::response::sse::Event::default()
                    .event("error")
                    .data(format!("Failed to create HTTP client: {}", e)));
                return;
            }
        };

        // Download the archive
        let response = match client.get(download_url).send().await {
            Ok(r) => r,
            Err(e) => {
                yield Ok(axum::response::sse::Event::default()
                    .event("error")
                    .data(format!("Failed to download: {}", e)));
                return;
            }
        };

        if !response.status().is_success() {
            yield Ok(axum::response::sse::Event::default()
                .event("error")
                .data(format!("Download failed: HTTP {}", response.status())));
            return;
        }

        let total_size = response.content_length().unwrap_or(0);

        // Download to a temp file
        let temp_path = std::path::Path::new(&lib_dir).join("onnxruntime_download.tmp");
        let mut file = match tokio::fs::File::create(&temp_path).await {
            Ok(f) => f,
            Err(e) => {
                yield Ok(axum::response::sse::Event::default()
                    .event("error")
                    .data(format!("Failed to create temp file: {}", e)));
                return;
            }
        };

        use tokio::io::AsyncWriteExt;
        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();
        use futures::StreamExt;

        let mut last_progress_time = std::time::Instant::now();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    if let Err(e) = file.write_all(&chunk).await {
                        yield Ok(axum::response::sse::Event::default()
                            .event("error")
                            .data(format!("Failed to write: {}", e)));
                        return;
                    }
                    downloaded += chunk.len() as u64;

                    if last_progress_time.elapsed() >= Duration::from_millis(100) {
                        last_progress_time = std::time::Instant::now();
                        yield Ok(axum::response::sse::Event::default()
                            .event("progress")
                            .data(serde_json::json!({
                                "status": "downloading",
                                "bytes_downloaded": downloaded,
                                "total_bytes": total_size
                            }).to_string()));
                    }
                }
                Err(e) => {
                    yield Ok(axum::response::sse::Event::default()
                        .event("error")
                        .data(format!("Download error: {}", e)));
                    return;
                }
            }
        }

        if let Err(e) = file.flush().await {
            yield Ok(axum::response::sse::Event::default()
                .event("error")
                .data(format!("Failed to flush: {}", e)));
            return;
        }
        drop(file);

        yield Ok(axum::response::sse::Event::default()
            .event("status")
            .data("Extracting ONNX Runtime..."));

        // Extract the archive (it's a .tgz file)
        let extract_result = tokio::task::spawn_blocking(move || {
            let file = std::fs::File::open(&temp_path)?;
            let decoder = flate2::read::GzDecoder::new(file);
            let mut archive = tar::Archive::new(decoder);

            // Extract and find the library file
            for entry in archive.entries()? {
                let mut entry = entry?;
                let path = entry.path()?;
                let path_str = path.to_string_lossy();

                // Look for the library file in the archive
                // Note: The archive contains a symlink (libonnxruntime.dylib) pointing to
                // the versioned file (libonnxruntime.1.22.0.dylib). We need to extract
                // the actual versioned file, not the symlink.
                let is_versioned_lib = if cfg!(target_os = "macos") {
                    // Match libonnxruntime.X.Y.Z.dylib
                    path_str.contains("libonnxruntime.") && path_str.ends_with(".dylib")
                        && !path_str.ends_with("/libonnxruntime.dylib")
                        && !path_str.contains(".dSYM")
                } else if cfg!(target_os = "linux") {
                    // Match libonnxruntime.so.X.Y.Z
                    path_str.contains("libonnxruntime.so.") && !path_str.ends_with("/libonnxruntime.so")
                } else {
                    // Windows: onnxruntime.dll (no versioning)
                    path_str.ends_with(lib_filename)
                };

                if is_versioned_lib {
                    let dest_path = std::path::Path::new(&lib_dir).join(lib_filename);
                    let mut dest_file = std::fs::File::create(&dest_path)?;
                    std::io::copy(&mut entry, &mut dest_file)?;
                    break;
                }
            }

            // Clean up temp file
            let _ = std::fs::remove_file(&temp_path);

            // Verify the library exists
            let final_path = std::path::Path::new(&lib_dir).join(lib_filename);
            if final_path.exists() {
                Ok(())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Library file not found in archive",
                ))
            }
        })
        .await;

        match extract_result {
            Ok(Ok(())) => {
                yield Ok(axum::response::sse::Event::default()
                    .event("complete")
                    .data("ONNX Runtime installed successfully"));
            }
            Ok(Err(e)) => {
                yield Ok(axum::response::sse::Event::default()
                    .event("error")
                    .data(format!("Extraction failed: {}", e)));
            }
            Err(e) => {
                yield Ok(axum::response::sse::Event::default()
                    .event("error")
                    .data(format!("Task failed: {}", e)));
            }
        }
    };

    Sse::new(stream)
}

// ============================================================================
// Model Management
// ============================================================================

// Model file URLs from HuggingFace
const MODEL_FILES: &[(&str, &str)] = &[
    (
        "encoder.onnx",
        "https://huggingface.co/altunenes/parakeet-rs/resolve/main/nemotron-speech-streaming-en-0.6b/encoder.onnx",
    ),
    (
        "encoder.onnx.data",
        "https://huggingface.co/altunenes/parakeet-rs/resolve/main/nemotron-speech-streaming-en-0.6b/encoder.onnx.data",
    ),
    (
        "decoder_joint.onnx",
        "https://huggingface.co/altunenes/parakeet-rs/resolve/main/nemotron-speech-streaming-en-0.6b/decoder_joint.onnx",
    ),
    (
        "tokenizer.model",
        "https://huggingface.co/altunenes/parakeet-rs/resolve/main/nemotron-speech-streaming-en-0.6b/tokenizer.model",
    ),
];

/// Get the model directory path for local transcription
pub fn get_nemotron_model_path() -> String {
    // Check environment variable first
    if let Ok(path) = std::env::var("NEMOTRON_MODEL_PATH") {
        return path;
    }

    // Default to models/nemotron in goose data directory
    goose::config::paths::Paths::in_data_dir("models/nemotron")
        .to_string_lossy()
        .to_string()
}

/// Get detailed model status
pub async fn get_model_status() -> Result<Json<serde_json::Value>, StatusCode> {
    let model_path = get_nemotron_model_path();
    let path = std::path::Path::new(&model_path);

    let mut files_status = Vec::new();
    let mut total_size: u64 = 0;
    let mut all_present = true;

    for (filename, _) in MODEL_FILES {
        let file_path = path.join(filename);
        if file_path.exists() {
            if let Ok(metadata) = std::fs::metadata(&file_path) {
                let size = metadata.len();
                total_size += size;
                files_status.push(serde_json::json!({
                    "name": filename,
                    "exists": true,
                    "size": size
                }));
            }
        } else {
            all_present = false;
            files_status.push(serde_json::json!({
                "name": filename,
                "exists": false,
                "size": 0
            }));
        }
    }

    Ok(Json(serde_json::json!({
        "installed": all_present,
        "path": model_path,
        "total_size": total_size,
        "files": files_status
    })))
}

/// Delete the local model files
pub async fn delete_model() -> Result<Json<serde_json::Value>, StatusCode> {
    let model_path = get_nemotron_model_path();
    let path = std::path::Path::new(&model_path);

    let mut deleted = Vec::new();
    let mut errors = Vec::new();

    for (filename, _) in MODEL_FILES {
        let file_path = path.join(filename);
        if file_path.exists() {
            match std::fs::remove_file(&file_path) {
                Ok(_) => deleted.push(filename.to_string()),
                Err(e) => errors.push(format!("{}: {}", filename, e)),
            }
        }
    }

    if errors.is_empty() {
        Ok(Json(serde_json::json!({
            "success": true,
            "deleted": deleted
        })))
    } else {
        Ok(Json(serde_json::json!({
            "success": false,
            "deleted": deleted,
            "errors": errors
        })))
    }
}

/// Download model files with SSE progress updates
#[allow(clippy::too_many_lines)]
pub async fn download_model() -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>>>
{
    let stream = async_stream::stream! {
        let model_path = get_nemotron_model_path();
        let path = std::path::Path::new(&model_path);

        // Create directory if it doesn't exist
        if let Err(e) = std::fs::create_dir_all(path) {
            yield Ok(axum::response::sse::Event::default()
                .event("error")
                .data(format!("Failed to create model directory: {}", e)));
            return;
        }

        yield Ok(axum::response::sse::Event::default()
            .event("status")
            .data("Starting download..."));

        let client = match Client::builder()
            .timeout(Duration::from_secs(3600))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                yield Ok(axum::response::sse::Event::default()
                    .event("error")
                    .data(format!("Failed to create HTTP client: {}", e)));
                return;
            }
        };

        let total_files = MODEL_FILES.len();

        for (index, (filename, url)) in MODEL_FILES.iter().enumerate() {
            let file_path = path.join(filename);

            yield Ok(axum::response::sse::Event::default()
                .event("progress")
                .data(serde_json::json!({
                    "file": filename,
                    "file_index": index,
                    "total_files": total_files,
                    "status": "starting",
                    "bytes_downloaded": 0,
                    "total_bytes": 0
                }).to_string()));

            // Start the download
            let response = match client.get(*url).send().await {
                Ok(r) => r,
                Err(e) => {
                    yield Ok(axum::response::sse::Event::default()
                        .event("error")
                        .data(format!("Failed to download {}: {}", filename, e)));
                    return;
                }
            };

            if !response.status().is_success() {
                yield Ok(axum::response::sse::Event::default()
                    .event("error")
                    .data(format!("Failed to download {}: HTTP {}", filename, response.status())));
                return;
            }

            let total_size = response.content_length().unwrap_or(0);

            // Create the file
            let mut file = match tokio::fs::File::create(&file_path).await {
                Ok(f) => f,
                Err(e) => {
                    yield Ok(axum::response::sse::Event::default()
                        .event("error")
                        .data(format!("Failed to create file {}: {}", filename, e)));
                    return;
                }
            };

            // Stream the download
            use tokio::io::AsyncWriteExt;
            let mut downloaded: u64 = 0;
            let mut stream = response.bytes_stream();
            use futures::StreamExt;

            let mut last_progress_time = std::time::Instant::now();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        if let Err(e) = file.write_all(&chunk).await {
                            yield Ok(axum::response::sse::Event::default()
                                .event("error")
                                .data(format!("Failed to write to {}: {}", filename, e)));
                            return;
                        }
                        downloaded += chunk.len() as u64;

                        // Send progress update every 100ms to avoid flooding
                        if last_progress_time.elapsed() >= Duration::from_millis(100) {
                            last_progress_time = std::time::Instant::now();
                            yield Ok(axum::response::sse::Event::default()
                                .event("progress")
                                .data(serde_json::json!({
                                    "file": filename,
                                    "file_index": index,
                                    "total_files": total_files,
                                    "status": "downloading",
                                    "bytes_downloaded": downloaded,
                                    "total_bytes": total_size
                                }).to_string()));
                        }
                    }
                    Err(e) => {
                        yield Ok(axum::response::sse::Event::default()
                            .event("error")
                            .data(format!("Download error for {}: {}", filename, e)));
                        return;
                    }
                }
            }

            // Flush and close the file
            if let Err(e) = file.flush().await {
                yield Ok(axum::response::sse::Event::default()
                    .event("error")
                    .data(format!("Failed to flush {}: {}", filename, e)));
                return;
            }

            yield Ok(axum::response::sse::Event::default()
                .event("progress")
                .data(serde_json::json!({
                    "file": filename,
                    "file_index": index,
                    "total_files": total_files,
                    "status": "complete",
                    "bytes_downloaded": downloaded,
                    "total_bytes": total_size
                }).to_string()));
        }

        yield Ok(axum::response::sse::Event::default()
            .event("complete")
            .data("All model files downloaded successfully"));
    };

    Sse::new(stream)
}

// ============================================================================
// Local Transcription
// ============================================================================

/// Initialize or get the local transcriber
async fn get_local_transcriber() -> Result<&'static std::sync::Mutex<Transcriber>, StatusCode> {
    // First ensure ONNX Runtime is initialized
    if let Err(e) = init_ort() {
        tracing::error!("ONNX Runtime not available: {}", e);
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    LOCAL_TRANSCRIBER
        .get_or_try_init(|| async {
            let model_path = get_nemotron_model_path();
            tracing::info!("Initializing local Nemotron transcriber from: {}", model_path);

            Transcriber::new(&model_path)
                .map(std::sync::Mutex::new)
                .map_err(|e| {
                    tracing::error!("Failed to initialize Nemotron transcriber: {:?}", e);
                    StatusCode::SERVICE_UNAVAILABLE
                })
        })
        .await
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)
}

/// Transcribe audio using local Nemotron model
///
/// Uses the local Nemotron 0.6B model for offline speech-to-text.
/// No API key required - runs entirely on-device.
pub async fn transcribe_local_handler(
    audio_bytes: Vec<u8>,
    mime_type: String,
) -> Result<String, StatusCode> {
    let transcriber = get_local_transcriber().await?;

    // Run transcription in a blocking task since the model inference is CPU-bound
    let text = tokio::task::spawn_blocking(move || {
        let mut guard = transcriber.lock().map_err(|e| {
            tracing::error!("Failed to lock transcriber: {:?}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        guard.transcribe(&audio_bytes, &mime_type).map_err(|e| {
            tracing::error!("Local transcription failed: {:?}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })
    })
    .await
    .map_err(|e| {
        tracing::error!("Transcription task panicked: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })??;

    Ok(text)
}
