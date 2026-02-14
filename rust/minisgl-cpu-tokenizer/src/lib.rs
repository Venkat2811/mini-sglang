use std::collections::HashMap;

use anyhow::Result;
use llm_tokenizer::{
    chat_template::ChatTemplateParams, Decoder, Encoder, HuggingFaceTokenizer, TokenizerTrait,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PromptInput {
    Text { text: String },
    Messages { messages: Vec<ChatMessage> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenizeRequest {
    pub uid: u64,
    pub prompt: PromptInput,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TokenizeOutput {
    pub uid: u64,
    pub input_ids: Vec<i32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DetokenizeRequest {
    pub uid: u64,
    pub next_token: i32,
    pub finished: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DetokenizeOutput {
    pub uid: u64,
    pub incremental_output: String,
    pub finished: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DecodeStatus {
    decoded_ids: Vec<u32>,
    decoded_str: String,
    read_offset: usize,
    surr_offset: usize,
    sent_offset_chars: usize,
}

impl DecodeStatus {
    fn new() -> Self {
        Self {
            decoded_ids: Vec::new(),
            decoded_str: String::new(),
            read_offset: 0,
            surr_offset: 0,
            sent_offset_chars: 0,
        }
    }
}

pub trait TokenizerBackend: Send + Sync {
    fn encode_one(&self, text: &str) -> Result<Vec<u32>>;
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>>;
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String>;
    fn decode_batch(&self, ids: &[Vec<u32>]) -> Result<Vec<String>>;
    fn eos_token_id(&self) -> Option<u32>;
}

pub struct HfTokenizerBackend {
    tokenizer: HuggingFaceTokenizer,
    eos_token_id: Option<u32>,
}

impl HfTokenizerBackend {
    pub fn from_model_path(model_path: &str) -> Result<Self> {
        Self::from_model_path_with_chat_template(model_path, None)
    }

    pub fn from_model_path_with_chat_template(
        model_path: &str,
        chat_template_path: Option<&str>,
    ) -> Result<Self> {
        let tokenizer_path = resolve_tokenizer_json_path(model_path);
        let tokenizer = HuggingFaceTokenizer::from_file_with_chat_template(
            &tokenizer_path,
            chat_template_path,
        )?;
        let eos_token_id = tokenizer
            .get_special_tokens()
            .eos_token
            .as_ref()
            .and_then(|eos| tokenizer.token_to_id(eos));

        Ok(Self {
            tokenizer,
            eos_token_id,
        })
    }
}

impl TokenizerBackend for HfTokenizerBackend {
    fn encode_one(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(text, true)?.token_ids().to_vec())
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        let encodings = self.tokenizer.encode_batch(texts, true)?;
        Ok(encodings
            .into_iter()
            .map(|encoding| encoding.token_ids().to_vec())
            .collect())
    }

    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        let json_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();
        self.tokenizer.apply_chat_template(
            &json_messages,
            ChatTemplateParams {
                add_generation_prompt: true,
                tools: None,
                documents: None,
                template_kwargs: None,
            },
        )
    }

    fn decode_batch(&self, ids: &[Vec<u32>]) -> Result<Vec<String>> {
        ids.iter()
            .map(|token_ids| self.tokenizer.decode(token_ids, false))
            .collect()
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }
}

pub struct TokenizeManager<B: TokenizerBackend> {
    backend: B,
}

impl<B: TokenizerBackend> TokenizeManager<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn tokenize(&self, requests: &[TokenizeRequest]) -> Result<Vec<TokenizeOutput>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let all_text = requests
            .iter()
            .all(|r| matches!(r.prompt, PromptInput::Text { .. }));
        if all_text {
            let mut texts: Vec<&str> = Vec::with_capacity(requests.len());
            for request in requests {
                if let PromptInput::Text { text } = &request.prompt {
                    texts.push(text.as_str());
                }
            }

            let encoded = self.backend.encode_batch(&texts)?;
            let outputs = requests
                .iter()
                .zip(encoded)
                .map(|(request, ids)| {
                    Ok(TokenizeOutput {
                        uid: request.uid,
                        input_ids: cast_u32_to_i32(ids)?,
                    })
                })
                .collect::<Result<Vec<TokenizeOutput>>>()?;
            return Ok(outputs);
        }

        let mut outputs = Vec::with_capacity(requests.len());
        for request in requests {
            let ids = match &request.prompt {
                PromptInput::Text { text } => self.backend.encode_one(text)?,
                PromptInput::Messages { messages } => {
                    let prompt = self.backend.apply_chat_template(messages)?;
                    self.backend.encode_one(&prompt)?
                }
            };

            outputs.push(TokenizeOutput {
                uid: request.uid,
                input_ids: cast_u32_to_i32(ids)?,
            });
        }
        Ok(outputs)
    }
}

pub struct DetokenizeManager<B: TokenizerBackend> {
    backend: B,
    decode_map: HashMap<u64, DecodeStatus>,
    eos_token_id: Option<u32>,
}

impl<B: TokenizerBackend> DetokenizeManager<B> {
    pub fn new(backend: B) -> Self {
        let eos_token_id = backend.eos_token_id();
        Self {
            backend,
            decode_map: HashMap::new(),
            eos_token_id,
        }
    }

    pub fn detokenize(&mut self, requests: &[DetokenizeRequest]) -> Result<Vec<DetokenizeOutput>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let mut read_ids = Vec::with_capacity(requests.len());
        let mut surr_ids = Vec::with_capacity(requests.len());

        for request in requests {
            let state = self
                .decode_map
                .entry(request.uid)
                .or_insert_with(DecodeStatus::new);
            let token = i32_to_u32(request.next_token)?;
            let is_final_eos = request.finished && self.eos_token_id == Some(token);
            if !is_final_eos {
                state.decoded_ids.push(token);
            }
            read_ids.push(state.decoded_ids[state.surr_offset..].to_vec());
            surr_ids.push(state.decoded_ids[state.surr_offset..state.read_offset].to_vec());
        }

        let read_texts = self.backend.decode_batch(&read_ids)?;
        let surr_texts = self.backend.decode_batch(&surr_ids)?;

        let mut outputs = Vec::with_capacity(requests.len());
        for ((request, read_text), surr_text) in requests.iter().zip(read_texts).zip(surr_texts) {
            let state = self
                .decode_map
                .get_mut(&request.uid)
                .ok_or_else(|| anyhow::anyhow!("missing decode status for uid={}", request.uid))?;

            let mut new_text = slice_from_char_idx(&read_text, surr_text.chars().count());

            let output_str = if !new_text.is_empty() && !new_text.ends_with('\u{FFFD}') {
                let mut output = String::with_capacity(state.decoded_str.len() + new_text.len());
                output.push_str(&state.decoded_str);
                output.push_str(&new_text);
                state.decoded_str = output.clone();
                state.surr_offset = state.read_offset;
                state.read_offset = state.decoded_ids.len();
                output
            } else {
                new_text = find_printable_text(&new_text);
                let mut output = String::with_capacity(state.decoded_str.len() + new_text.len());
                output.push_str(&state.decoded_str);
                output.push_str(&new_text);
                output
            };

            let incremental_output = slice_from_char_idx(&output_str, state.sent_offset_chars);
            state.sent_offset_chars = output_str.chars().count();
            outputs.push(DetokenizeOutput {
                uid: request.uid,
                incremental_output,
                finished: request.finished,
            });

            if request.finished {
                self.decode_map.remove(&request.uid);
            }
        }

        Ok(outputs)
    }

    pub fn active_sequences(&self) -> usize {
        self.decode_map.len()
    }
}

fn is_chinese_char(cp: u32) -> bool {
    ((0x4E00..=0x9FFF).contains(&cp))
        || ((0x3400..=0x4DBF).contains(&cp))
        || ((0x20000..=0x2A6DF).contains(&cp))
        || ((0x2A700..=0x2B73F).contains(&cp))
        || ((0x2B740..=0x2B81F).contains(&cp))
        || ((0x2B820..=0x2CEAF).contains(&cp))
        || ((0xF900..=0xFAFF).contains(&cp))
        || ((0x2F800..=0x2FA1F).contains(&cp))
}

fn find_printable_text(text: &str) -> String {
    if text.ends_with('\n') {
        return text.to_string();
    }

    let mut chars = text.chars().rev();
    let last = chars.next();
    let penultimate = chars.next();

    if let Some(c) = last {
        if is_chinese_char(c as u32) {
            return text.to_string();
        }
    }

    if let Some(c) = penultimate {
        if is_chinese_char(c as u32) {
            return text
                .chars()
                .take(text.chars().count().saturating_sub(1))
                .collect();
        }
    }

    match text.rfind(' ') {
        Some(idx) => text[..=idx].to_string(),
        None => String::new(),
    }
}

fn slice_from_char_idx(text: &str, start_char_idx: usize) -> String {
    if start_char_idx == 0 {
        return text.to_string();
    }
    let byte_idx = text
        .char_indices()
        .nth(start_char_idx)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len());
    text[byte_idx..].to_string()
}

fn cast_u32_to_i32(ids: Vec<u32>) -> Result<Vec<i32>> {
    if cfg!(debug_assertions) {
        if let Some(bad) = ids.iter().find(|id| **id > i32::MAX as u32) {
            return Err(anyhow::anyhow!("token id {} does not fit in i32", bad));
        }
    }
    Ok(ids.into_iter().map(|id| id as i32).collect())
}

fn i32_to_u32(id: i32) -> Result<u32> {
    u32::try_from(id).map_err(|_| anyhow::anyhow!("negative token id {}", id))
}

fn resolve_tokenizer_json_path(model_path: &str) -> String {
    let path = std::path::Path::new(model_path);
    if path.is_file() {
        return model_path.to_string();
    }
    if path.is_dir() {
        return path.join("tokenizer.json").to_string_lossy().to_string();
    }
    model_path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_printable_text_keeps_cjk_chars() {
        assert_eq!(find_printable_text("你好"), "你好");
        assert_eq!(find_printable_text("你a"), "你");
    }

    #[test]
    fn find_printable_text_keeps_word_boundary() {
        assert_eq!(find_printable_text("hello world"), "hello ");
        assert_eq!(find_printable_text("hello"), "");
    }

    #[test]
    fn slice_from_char_idx_handles_multibyte() {
        assert_eq!(slice_from_char_idx("a你b", 1), "你b");
        assert_eq!(slice_from_char_idx("a你b", 2), "b");
    }

    #[derive(Clone, Default)]
    struct FakeBackend;

    impl TokenizerBackend for FakeBackend {
        fn encode_one(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
            Ok(texts
                .iter()
                .map(|t| t.bytes().map(u32::from).collect::<Vec<u32>>())
                .collect())
        }

        fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
            Ok(messages
                .iter()
                .map(|m| format!("{}:{}", m.role, m.content))
                .collect::<Vec<String>>()
                .join("\n"))
        }

        fn decode_batch(&self, ids: &[Vec<u32>]) -> Result<Vec<String>> {
            let mut out = Vec::with_capacity(ids.len());
            for item in ids {
                let mut s = String::new();
                for &id in item {
                    s.push(char::from_u32(id).unwrap_or('\u{FFFD}'));
                }
                out.push(s);
            }
            Ok(out)
        }

        fn eos_token_id(&self) -> Option<u32> {
            Some(0)
        }
    }

    fn python_tokenize_oracle<B: TokenizerBackend>(
        backend: &B,
        requests: &[TokenizeRequest],
    ) -> Vec<TokenizeOutput> {
        let mut out = Vec::with_capacity(requests.len());
        for request in requests {
            let ids = match &request.prompt {
                PromptInput::Text { text } => backend.encode_one(text).expect("encode_one"),
                PromptInput::Messages { messages } => {
                    let prompt = backend
                        .apply_chat_template(messages)
                        .expect("apply_chat_template");
                    backend.encode_one(&prompt).expect("encode_one")
                }
            };
            out.push(TokenizeOutput {
                uid: request.uid,
                input_ids: cast_u32_to_i32(ids).expect("cast_u32_to_i32"),
            });
        }
        out
    }

    struct PythonDetokenizeOracle {
        decode_map: HashMap<u64, DecodeStatus>,
        eos_token_id: Option<u32>,
    }

    impl PythonDetokenizeOracle {
        fn new(eos_token_id: Option<u32>) -> Self {
            Self {
                decode_map: HashMap::new(),
                eos_token_id,
            }
        }

        fn detokenize<B: TokenizerBackend>(
            &mut self,
            backend: &B,
            requests: &[DetokenizeRequest],
        ) -> Vec<String> {
            let mut read_ids = Vec::with_capacity(requests.len());
            let mut surr_ids = Vec::with_capacity(requests.len());
            for request in requests {
                let state = self
                    .decode_map
                    .entry(request.uid)
                    .or_insert_with(DecodeStatus::new);
                let token = i32_to_u32(request.next_token).expect("i32_to_u32");
                let is_final_eos = request.finished && self.eos_token_id == Some(token);
                if !is_final_eos {
                    state.decoded_ids.push(token);
                }
                read_ids.push(state.decoded_ids[state.surr_offset..].to_vec());
                surr_ids.push(state.decoded_ids[state.surr_offset..state.read_offset].to_vec());
            }

            let read_texts = backend.decode_batch(&read_ids).expect("decode_batch");
            let surr_texts = backend.decode_batch(&surr_ids).expect("decode_batch");
            let mut out = Vec::with_capacity(requests.len());
            for ((request, read_str), surr_str) in requests.iter().zip(read_texts).zip(surr_texts) {
                let state = self.decode_map.get_mut(&request.uid).expect("decode state");
                let mut new_text = slice_from_char_idx(&read_str, surr_str.chars().count());
                let output_str = if !new_text.is_empty() && !new_text.ends_with('\u{FFFD}') {
                    let mut output = String::with_capacity(state.decoded_str.len() + new_text.len());
                    output.push_str(&state.decoded_str);
                    output.push_str(&new_text);
                    state.decoded_str = output.clone();
                    state.surr_offset = state.read_offset;
                    state.read_offset = state.decoded_ids.len();
                    output
                } else {
                    new_text = find_printable_text(&new_text);
                    let mut output = String::with_capacity(state.decoded_str.len() + new_text.len());
                    output.push_str(&state.decoded_str);
                    output.push_str(&new_text);
                    output
                };

                let incremental_output = slice_from_char_idx(&output_str, state.sent_offset_chars);
                state.sent_offset_chars = output_str.chars().count();
                out.push(incremental_output);
                if request.finished {
                    self.decode_map.remove(&request.uid);
                }
            }
            out
        }

        fn active_sequences(&self) -> usize {
            self.decode_map.len()
        }
    }

    #[test]
    fn tokenize_matches_python_oracle_for_text_and_chat_prompts() {
        let backend = FakeBackend;
        let mgr = TokenizeManager::new(backend.clone());
        let requests = vec![
            TokenizeRequest {
                uid: 11,
                prompt: PromptInput::Text {
                    text: "alpha beta".to_string(),
                },
            },
            TokenizeRequest {
                uid: 12,
                prompt: PromptInput::Messages {
                    messages: vec![
                        ChatMessage {
                            role: "system".to_string(),
                            content: "You are concise.".to_string(),
                        },
                        ChatMessage {
                            role: "user".to_string(),
                            content: "Say hi".to_string(),
                        },
                    ],
                },
            },
            TokenizeRequest {
                uid: 13,
                prompt: PromptInput::Text {
                    text: "你好".to_string(),
                },
            },
        ];

        let expected = python_tokenize_oracle(&backend, &requests);
        let actual = mgr.tokenize(&requests).expect("tokenize");
        assert_eq!(actual, expected);
    }

    #[test]
    fn detokenize_matches_python_oracle_for_interleaved_multilingual_streams() {
        let backend = FakeBackend;
        let mut rust_mgr = DetokenizeManager::new(backend.clone());
        let mut python_oracle = PythonDetokenizeOracle::new(backend.eos_token_id());

        let steps = vec![
            vec![
                DetokenizeRequest {
                    uid: 1,
                    next_token: '你' as i32,
                    finished: false,
                },
                DetokenizeRequest {
                    uid: 2,
                    next_token: 'h' as i32,
                    finished: false,
                },
            ],
            vec![
                DetokenizeRequest {
                    uid: 1,
                    next_token: 0xD800,
                    finished: false,
                },
                DetokenizeRequest {
                    uid: 2,
                    next_token: 'i' as i32,
                    finished: false,
                },
            ],
            vec![
                DetokenizeRequest {
                    uid: 1,
                    next_token: '好' as i32,
                    finished: false,
                },
                DetokenizeRequest {
                    uid: 2,
                    next_token: ' ' as i32,
                    finished: false,
                },
            ],
            vec![
                DetokenizeRequest {
                    uid: 1,
                    next_token: 0,
                    finished: true,
                },
                DetokenizeRequest {
                    uid: 2,
                    next_token: 0,
                    finished: true,
                },
            ],
        ];

        for requests in steps {
            let expected = python_oracle.detokenize(&backend, &requests);
            let actual = rust_mgr.detokenize(&requests).expect("detokenize");
            let actual_text = actual
                .into_iter()
                .map(|item| item.incremental_output)
                .collect::<Vec<String>>();
            assert_eq!(actual_text, expected);
            assert_eq!(rust_mgr.active_sequences(), python_oracle.active_sequences());
        }
    }

    #[test]
    fn tokenize_plain_text_batch_is_supported() {
        let mgr = TokenizeManager::new(FakeBackend);
        let requests = vec![
            TokenizeRequest {
                uid: 1,
                prompt: PromptInput::Text {
                    text: "ab".to_string(),
                },
            },
            TokenizeRequest {
                uid: 2,
                prompt: PromptInput::Text {
                    text: "cd".to_string(),
                },
            },
        ];
        let out = mgr.tokenize(&requests).expect("tokenize");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].input_ids, vec![97, 98]);
        assert_eq!(out[1].input_ids, vec![99, 100]);
    }

    #[test]
    fn detokenize_incremental_streaming_flushes_on_word_boundaries() {
        let mut mgr = DetokenizeManager::new(FakeBackend);
        let out_0 = mgr
            .detokenize(&[DetokenizeRequest {
                uid: 10,
                next_token: 'h' as i32,
                finished: false,
            }])
            .expect("detokenize step 0");
        let out_1 = mgr
            .detokenize(&[DetokenizeRequest {
                uid: 10,
                next_token: 'i' as i32,
                finished: false,
            }])
            .expect("detokenize step 1");
        let out_2 = mgr
            .detokenize(&[DetokenizeRequest {
                uid: 10,
                next_token: ' ' as i32,
                finished: false,
            }])
            .expect("detokenize step 2");
        let out_3 = mgr
            .detokenize(&[DetokenizeRequest {
                uid: 10,
                next_token: 0,
                finished: true,
            }])
            .expect("detokenize step 3");

        assert_eq!(out_0[0].incremental_output, "h");
        assert_eq!(out_1[0].incremental_output, "i");
        assert_eq!(out_2[0].incremental_output, " ");
        assert_eq!(out_3[0].incremental_output, "");
        assert_eq!(mgr.active_sequences(), 0);
    }

    #[test]
    fn detokenize_streaming_handles_cjk_chars_without_byte_offset_corruption() {
        let mut mgr = DetokenizeManager::new(FakeBackend);
        let out_0 = mgr
            .detokenize(&[DetokenizeRequest {
                uid: 77,
                next_token: '你' as i32,
                finished: false,
            }])
            .expect("detokenize cjk step 0");
        let out_1 = mgr
            .detokenize(&[DetokenizeRequest {
                uid: 77,
                next_token: '好' as i32,
                finished: false,
            }])
            .expect("detokenize cjk step 1");
        let out_2 = mgr
            .detokenize(&[DetokenizeRequest {
                uid: 77,
                next_token: 0,
                finished: true,
            }])
            .expect("detokenize cjk step 2");

        assert_eq!(out_0[0].incremental_output, "你");
        assert_eq!(out_1[0].incremental_output, "好");
        assert_eq!(out_2[0].incremental_output, "");
        assert_eq!(mgr.active_sequences(), 0);
    }
}
