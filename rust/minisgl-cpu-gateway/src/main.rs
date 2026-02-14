use std::{env, net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tokio::signal;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Clone)]
struct GatewayState {
    model_id: Arc<str>,
    workers: Arc<[String]>,
    http_client: reqwest::Client,
}

#[derive(Clone, Debug)]
struct GatewayConfig {
    listen_addr: SocketAddr,
    model_id: Arc<str>,
    workers: Vec<String>,
    request_timeout_ms: u64,
}

impl GatewayConfig {
    fn from_env() -> Self {
        let listen_addr = env::var("MINISGL_GATEWAY_ADDR")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 3030)));
        let model_id = Arc::<str>::from(
            env::var("MINISGL_GATEWAY_MODEL_ID")
                .unwrap_or_else(|_| "mini-sgl-placeholder".to_string()),
        );
        let workers = env::var("MINISGL_GATEWAY_WORKERS")
            .map(|s| Self::parse_workers(&s))
            .unwrap_or_default();
        let request_timeout_ms = env::var("MINISGL_GATEWAY_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3_000_u64);
        Self {
            listen_addr,
            model_id,
            workers,
            request_timeout_ms,
        }
    }

    fn parse_workers(raw: &str) -> Vec<String> {
        raw.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect()
    }
}

impl GatewayState {
    fn from_config(config: &GatewayConfig) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_millis(config.request_timeout_ms))
            .build()
            .expect("build reqwest client");
        Self {
            model_id: config.model_id.clone(),
            workers: Arc::from(config.workers.clone()),
            http_client,
        }
    }
}

fn build_app(state: GatewayState) -> Router {
    Router::new()
        .route("/liveness", get(liveness))
        .route("/readiness", get(readiness))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

#[tokio::main]
async fn main() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let config = GatewayConfig::from_env();
    let state = GatewayState::from_config(&config);
    let app = build_app(state);

    info!(
        addr = %config.listen_addr,
        model_id = %config.model_id,
        workers = config.workers.len(),
        "starting minisgl-cpu-gateway"
    );

    let listener = tokio::net::TcpListener::bind(config.listen_addr)
        .await
        .expect("bind minisgl-cpu-gateway listener");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("run minisgl-cpu-gateway server");
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(err) = signal::ctrl_c().await {
            warn!(error = %err, "ctrl_c handler error");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                let _ = sig.recv().await;
            }
            Err(err) => {
                warn!(error = %err, "failed to install SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("shutdown signal received");
}

async fn liveness() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

fn worker_health_url(worker: &str) -> String {
    format!("{}/healthz", worker.trim_end_matches('/'))
}

fn worker_ready_url(worker: &str) -> String {
    format!("{}/v1/models", worker.trim_end_matches('/'))
}

fn worker_chat_url(worker: &str) -> String {
    format!("{}/v1/chat/completions", worker.trim_end_matches('/'))
}

async fn worker_is_healthy(state: &GatewayState, worker: &str) -> bool {
    let probe_urls = [worker_health_url(worker), worker_ready_url(worker)];
    for url in probe_urls {
        match state.http_client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => return true,
            Ok(_) => {}
            Err(err) => {
                warn!(worker = %worker, error = %err, "worker health probe failed");
            }
        }
    }
    false
}

async fn readiness(State(state): State<GatewayState>) -> Response {
    let total_workers = state.workers.len();
    if total_workers == 0 {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "ready": false,
                "healthy_workers": 0,
                "total_workers": 0,
                "reason": "no_workers_configured",
            })),
        )
            .into_response();
    }

    let mut healthy_workers = 0_usize;
    for worker in state.workers.iter() {
        if worker_is_healthy(&state, worker).await {
            healthy_workers += 1;
        }
    }
    let ready = healthy_workers > 0;
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(json!({
            "ready": ready,
            "healthy_workers": healthy_workers,
            "total_workers": total_workers,
        })),
    )
        .into_response()
}

async fn list_models(State(state): State<GatewayState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_id.as_ref(),
            "object": "model",
            "owned_by": "mini-sglang",
        }]
    }))
}

async fn chat_completions(State(state): State<GatewayState>, Json(body): Json<Value>) -> Response {
    if state.workers.is_empty() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": {
                    "message": "no gateway workers configured",
                    "type": "unavailable",
                    "code": "minisgl_cpu_gateway_no_workers",
                }
            })),
        )
            .into_response();
    }

    for worker in state.workers.iter() {
        let url = worker_chat_url(worker);
        match state.http_client.post(&url).json(&body).send().await {
            Ok(resp) => {
                let status =
                    StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
                let content_type = resp
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("application/json")
                    .to_string();
                match resp.bytes().await {
                    Ok(bytes) => {
                        if let Ok(payload) = serde_json::from_slice::<Value>(&bytes) {
                            return (status, Json(payload)).into_response();
                        }
                        let mut builder = Response::builder().status(status);
                        if let Ok(value) = axum::http::HeaderValue::from_str(&content_type) {
                            builder = builder.header(axum::http::header::CONTENT_TYPE, value);
                        }
                        return match builder.body(Body::from(bytes.to_vec())) {
                            Ok(response) => response,
                            Err(_) => (
                                StatusCode::BAD_GATEWAY,
                                Json(json!({
                                    "error": {
                                        "message": "gateway failed to forward upstream payload",
                                        "type": "bad_gateway",
                                        "code": "minisgl_cpu_gateway_forward_error",
                                    }
                                })),
                            )
                                .into_response(),
                        };
                    }
                    Err(err) => {
                        warn!(worker = %worker, error = %err, "chat pass-through read failed");
                    }
                }
            }
            Err(err) => {
                warn!(worker = %worker, error = %err, "chat pass-through request failed");
            }
        }
    }

    (
        StatusCode::BAD_GATEWAY,
        Json(json!({
            "error": {
                "message": "all configured workers are unreachable",
                "type": "bad_gateway",
                "code": "minisgl_cpu_gateway_all_workers_unreachable",
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use tower::ServiceExt;

    fn test_state(workers: Vec<String>) -> GatewayState {
        let config = GatewayConfig {
            listen_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
            model_id: Arc::from("test-model"),
            workers,
            request_timeout_ms: 500,
        };
        GatewayState::from_config(&config)
    }

    async fn spawn_mock_worker(healthy: bool) -> (String, tokio::task::JoinHandle<()>) {
        let health_status = if healthy {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        };
        let app = Router::new()
            .route(
                "/healthz",
                get(move || async move { (health_status, "ok") }),
            )
            .route(
                "/v1/chat/completions",
                post(|| async {
                    Json(json!({ "id": "mock", "object": "chat.completion", "choices": [] }))
                }),
            );
        let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .expect("bind mock worker");
        let addr = listener.local_addr().expect("local addr");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), handle)
    }

    async fn spawn_mock_stream_worker() -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new()
            .route("/v1/models", get(|| async { Json(json!({"ok": true})) }))
            .route(
                "/v1/chat/completions",
                post(|| async {
                    (
                        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                        "data: {\"id\":\"mock-stream\"}\n\ndata: [DONE]\n\n",
                    )
                }),
            );
        let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .expect("bind mock stream worker");
        let addr = listener.local_addr().expect("local addr");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), handle)
    }

    #[tokio::test]
    async fn liveness_endpoint_returns_ok() {
        let app = build_app(test_state(vec![]));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/liveness")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn readiness_is_unavailable_when_no_workers_configured() {
        let app = build_app(test_state(vec![]));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/readiness")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn readiness_is_ok_when_at_least_one_worker_is_healthy() {
        let (worker, handle) = spawn_mock_worker(true).await;
        let app = build_app(test_state(vec![worker]));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/readiness")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        handle.abort();
    }

    #[tokio::test]
    async fn models_endpoint_returns_model_id() {
        let app = build_app(test_state(vec![]));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        let payload: Value = serde_json::from_slice(&body).expect("parse json");
        assert_eq!(payload["data"][0]["id"], "test-model");
    }

    #[tokio::test]
    async fn chat_completions_passthrough_returns_worker_response() {
        let (worker, handle) = spawn_mock_worker(true).await;
        let app = build_app(test_state(vec![worker]));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test-model","messages":[{"role":"user","content":"hello"}]}"#,
            ))
            .expect("request");
        let response = app.oneshot(request).await.expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        let payload: Value = serde_json::from_slice(&body).expect("parse json");
        assert_eq!(payload["id"], "mock");
        handle.abort();
    }

    #[tokio::test]
    async fn chat_completions_passthrough_supports_streaming_payloads() {
        let (worker, handle) = spawn_mock_stream_worker().await;
        let app = build_app(test_state(vec![worker]));
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test-model","messages":[{"role":"user","content":"hello"}]}"#,
            ))
            .expect("request");
        let response = app.oneshot(request).await.expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or(""),
            "text/event-stream"
        );
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        let payload = String::from_utf8(body.to_vec()).expect("utf8");
        assert!(payload.contains("mock-stream"));
        handle.abort();
    }
}
