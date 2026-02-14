use std::{net::SocketAddr, sync::Arc};

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use openai_protocol::chat::ChatCompletionRequest;
use serde_json::json;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Clone)]
struct GatewayState {
    model_id: Arc<str>,
}

#[tokio::main]
async fn main() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let state = GatewayState {
        model_id: Arc::from("mini-sgl-placeholder"),
    };
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3030));
    info!(%addr, "starting minisgl-cpu-gateway placeholder");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind minisgl-cpu-gateway listener");
    axum::serve(listener, app)
        .await
        .expect("run minisgl-cpu-gateway server");
}

async fn healthz() -> &'static str {
    "ok"
}

async fn list_models(State(state): State<GatewayState>) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_id.as_ref(),
            "object": "model",
            "owned_by": "mini-sglang",
        }]
    }))
}

async fn chat_completions(
    State(state): State<GatewayState>,
    Json(_body): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": {
                "message": "chat/completions is scaffolded but not wired to engine yet",
                "type": "not_implemented",
                "code": "minisgl_cpu_gateway_stub",
                "model": state.model_id.as_ref(),
            }
        })),
    )
}
