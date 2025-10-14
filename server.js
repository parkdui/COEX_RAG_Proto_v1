// COEX RAG 워킹 프로토타입 (HyperCLOVAX HCX-005 전용)
// - CSV → 임베딩 저장 → 질의 임베딩 → 로컬 시맨틱 검색 → CLOVA Chat Completions

// ─── STT (CLOVA Speech Recognition) ──────────────────────────────
// const multer = require("multer");
// const upload = multer({ dest: "uploads/" }); // 업로드된 음성 임시 저장
// const request = require("request");

//v.1.2 updated

const fs = require("fs");
const path = require("path");
const express = require("express");
const cors = require("cors");
// const csv = require("csv-parser");
require("dotenv").config();
const { google } = require("googleapis");

const ALLOWED = [
  "http://localhost:3000",
  "https://coex-rag-proto-v1.vercel.app",
  "https://*.ngrok-free.app", // ngrok 쓰면
  "https://coex-backend.onrender.com", // Render/Railway 주소(있다면)
];

const http = require("http");
const { Server } = require("socket.io");
const app = express();
app.use(
  require("cors")({
    origin: (origin, cb) => cb(null, true), // 데모용: 임시 전체 허용
    credentials: true,
  })
);
app.use(
  cors({
    origin: (origin, cb) => cb(null, true), // 데모용: 임시로 모두 허용
    credentials: true,
  })
);
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));
// app.use(express.static("public"));
// app.use("/LLM", express.static("LLM"));

const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: (origin, cb) => cb(null, true),
    methods: ["GET", "POST"],
    credentials: true,
  },
});

// ========== ENV 헬퍼 (인라인 주석/공백 제거) ==========
const getEnv = (k, d = "") => {
  const v = process.env[k];
  if (!v) return d;
  // [수정] private key의 \n을 실제 개행 문자로 변환
  if (k === "GOOGLE_PRIVATE_KEY") {
    return v.replace(/\\n/g, "\n");
  }
  return String(v).split("#")[0].trim();
};

// ========== CSR & Voice ENV ==========
const CLOVA_SPEECH_ID = getEnv("CLOVA_SPEECH_CLIENT_ID");
const CLOVA_SPEECH_SECRET = getEnv("CLOVA_SPEECH_CLIENT_SECRET");

// ========== ENV 로드 & 자동 보정 ==========
const APP_ID = getEnv("APP_ID", "testapp"); // testapp | serviceapp
const TOP_K = parseInt(getEnv("TOP_K", "3"), 10);

// 1) Embedding/Segmentation BASE
let HLX_BASE = getEnv(
  "HYPERCLOVAX_API_BASE",
  "https://clovastudio.apigw.ntruss.com"
);
const HLX_KEY = getEnv("HYPERCLOVAX_API_KEY");
const EMB_MODEL = getEnv("HYPERCLOVAX_EMBED_MODEL", "clir-emb-dolphin");

// stream 도메인이면 apigw로 교체
if (/clovastudio\.stream\.ntruss\.com/.test(HLX_BASE)) {
  HLX_BASE = HLX_BASE.replace(
    "clovastudio.stream.ntruss.com",
    "clovastudio.apigw.ntruss.com"
  );
}
// /testapp|/serviceapp 경로 없으면 붙이기
if (!/\/(testapp|serviceapp)(\/|$)/.test(HLX_BASE)) {
  HLX_BASE = HLX_BASE.replace(/\/$/, "") + "/" + APP_ID;
}

// 2) Chat BASE
let CLOVA_BASE = getEnv(
  "CLOVA_API_BASE",
  "https://clovastudio.apigw.ntruss.com"
);

// [추가] /testapp|/serviceapp 경로 없으면 붙이기 (CLOVA_BASE에도 동일하게 적용)
if (!/\/(testapp|serviceapp)(\/|$)/.test(CLOVA_BASE)) {
  CLOVA_BASE = CLOVA_BASE.replace(/\/$/, "") + "/" + APP_ID;
}
const CLOVA_KEY = getEnv("CLOVA_API_KEY");
const CLOVA_MODEL = getEnv("CLOVA_MODEL", "HCX-005");

// Google Sheets ENV 로드
const GOOGLE_SHEET_ID = getEnv("GOOGLE_SHEET_ID");
const GOOGLE_SHEET_RANGE = getEnv("GOOGLE_SHEET_RANGE");
const GOOGLE_SERVICE_ACCOUNT_EMAIL = getEnv("GOOGLE_SERVICE_ACCOUNT_EMAIL");
const GOOGLE_PRIVATE_KEY = getEnv("GOOGLE_PRIVATE_KEY");

// Google Sheets 로깅용 ENV
const LOG_GOOGLE_SHEET_ID = getEnv("LOG_GOOGLE_SHEET_ID");
const LOG_GOOGLE_SHEET_NAME = getEnv("LOG_GOOGLE_SHEET_NAME"); // "Sheet1"

// 파일 경로
// const DATA_CSV = path.join(__dirname, "data", "event_lists.csv");
const VECTORS_JSON = path.join(__dirname, "data", "vectors.json");
const systemPrompt = fs.readFileSync(
  path.join(__dirname, "public", "LLM", "system_prompt.txt"),
  "utf8"
);

// --- systemPrompt 기본값 저장 (fallback) ---
let defaultSystemPrompt = systemPrompt;

// ---- Conversation Memory ----
const chatHistories = new Map();
const chatLogs = new Map();
const MAX_HISTORY = parseInt(getEnv("MAX_HISTORY", "20"), 10);
function getCid(req) {
  // 우선순위: body.conversationId > 헤더 > (fallback) ip+UA
  return (
    (req.body && req.body.conversationId) ||
    req.get("x-conversation-id") ||
    `${req.ip}:${(req.get("user-agent") || "").slice(0, 40)}`
  );
}

function pushHistory(socketId, role, content) {
  const arr = chatHistories.get(socketId) || [];
  arr.push({ role, content });
  if (arr.length > MAX_HISTORY) arr.splice(0, arr.length - MAX_HISTORY);
  chatHistories.set(socketId, arr);
}

// ====== 유틸: 코사인 유사도 ======
function cosineSim(a, b) {
  let dot = 0,
    na = 0,
    nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ==== Token counters ====
const TOKENS = {
  embed_input: 0,
  embed_calls: 0,
  chat_input: 0,
  chat_output: 0,
  chat_total: 0,
  chat_calls: 0,
};

function logTokenSummary(tag = "") {
  console.log(
    `🧮 [TOKENS${tag ? " " + tag : ""}] ` +
      `EMB in=${TOKENS.embed_input} (calls=${TOKENS.embed_calls}) | ` +
      `CHAT in=${TOKENS.chat_input} out=${TOKENS.chat_output} total=${TOKENS.chat_total} ` +
      `(calls=${TOKENS.chat_calls})`
  );
}

// ====== HyperCLOVAX Embedding API (보강판) ======
async function embedText(text) {
  if (!text || !text.trim()) throw new Error("empty text for embedding");

  const url = `${HLX_BASE}/v1/api-tools/embedding/${EMB_MODEL}`;
  const headers = {
    Authorization: `Bearer ${HLX_KEY}`,
    "Content-Type": "application/json",
    "X-NCP-CLOVASTUDIO-REQUEST-ID": `emb-${Date.now()}-${Math.random()}`,
  };

  // v1
  let res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({ text }),
  });

  // 4xx면 v2
  if (!res.ok && res.status >= 400 && res.status < 500) {
    res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify({ texts: [text] }),
    });
  }

  const raw = await res.text();
  let json;
  try {
    json = JSON.parse(raw);
  } catch {
    throw new Error(`Embedding invalid JSON: ${raw.slice(0, 300)}`);
  }

  const codeRaw = json?.status?.code ?? json?.code;
  const isOk = codeRaw === 20000 || codeRaw === "20000" || codeRaw == null;
  if (!isOk) {
    const msg = json?.status?.message || json?.message || "(no message)";
    throw new Error(`Embedding API status=${codeRaw} message=${msg}`);
  }

  // ----- add: embedding token usage logging -----
  const embUsage = json?.result?.usage ?? json?.usage ?? {};

  const embInput =
    Number(
      json?.result?.inputTokens ??
        json?.inputTokens ??
        embUsage.inputTokens ??
        0
    ) || 0;

  TOKENS.embed_input += embInput;
  TOKENS.embed_calls += 1;

  if (process.env.LOG_TOKENS === "1") {
    console.log(
      `📦 [EMB] inputTokens=${embInput} (acc=${TOKENS.embed_input}, calls=${TOKENS.embed_calls})`
    );
  }

  const emb = extractEmbedding(json);
  if (!emb) {
    throw new Error("Embedding response missing vector");
  }
  return emb;
}

function extractEmbedding(json) {
  const cands = [
    json?.result?.embedding, // ← 문서 예시
    json?.embedding,
    json?.result?.embeddings?.[0],
    json?.embeddings?.[0],
    json?.result?.embeddings?.[0]?.values,
    json?.result?.embeddings?.[0]?.vector,
    json?.embeddings?.[0]?.values,
    json?.embeddings?.[0]?.vector,
  ];
  for (const c of cands) {
    if (!c) continue;
    if (Array.isArray(c) && typeof c[0] === "number") return c;
    if (Array.isArray(c?.values) && typeof c.values[0] === "number")
      return c.values;
    if (Array.isArray(c?.vector) && typeof c.vector[0] === "number")
      return c.vector;
  }
  return null;
}

// ====== (옵션) 세그멘테이션 ======
async function segmentText(text) {
  const url = `${HLX_BASE}/v1/api-tools/segmentation`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${HLX_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      alpha: -100, // 자동 분할
      segCnt: -1, // 제한 없음
      postProcess: true,
      postProcessMaxSize: 1000,
      postProcessMinSize: 300,
    }),
  });
  if (!res.ok)
    throw new Error(
      `Segmentation failed ${res.status}: ${await res.text().catch(() => "")}`
    );
  const json = await res.json();
  return Array.isArray(json?.segments) ? json.segments : [text];
}

// ====== CLOVA Chat Completions v3 (non-stream) ======
async function callClovaChat(messages, opts = {}) {
  const url = `${CLOVA_BASE}/v3/chat-completions/${CLOVA_MODEL}`;

  // ✅ 메시지 포맷 변환
  const wrappedMessages = messages.map((m) => ({
    role: m.role,
    content: [{ type: "text", text: m.content }],
  }));

  const body = {
    messages: wrappedMessages,
    temperature: opts.temperature ?? 0.3,
    topP: opts.topP ?? 0.8,
    topK: opts.topK ?? 0,
    maxTokens: opts.maxTokens ?? 700,
    repeatPenalty: opts.repeatPenalty ?? 1.1,
    stop: [],
  };
  // 🟢 여기서 요청 바디 전체 로그 찍기
  console.log("📝 [CLOVA Chat Request Body]:", JSON.stringify(body, null, 2));

  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${CLOVA_KEY}`,
      "Content-Type": "application/json; charset=utf-8",
      "X-NCP-CLOVASTUDIO-REQUEST-ID": `req-${Date.now()}`,
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok)
    throw new Error(
      `CLOVA chat failed ${res.status}: ${await res.text().catch(() => "")}`
    );
  const json = await res.json();

  // ----- add: chat token usage logging -----
  const chatUsage =
    json?.result?.usage || // result 안에 들어오는 경우
    json?.usage ||
    {}; // 최상위에 오는 경우

  const chatIn = Number(chatUsage.promptTokens ?? 0);
  const chatOut = Number(chatUsage.completionTokens ?? 0);
  const chatTotal = Number(chatUsage.totalTokens ?? chatIn + chatOut);

  TOKENS.chat_input += chatIn;
  TOKENS.chat_output += chatOut;
  TOKENS.chat_total += chatTotal;
  TOKENS.chat_calls += 1;

  if (process.env.LOG_TOKENS === "1") {
    console.log(
      `💬 [CHAT] in=${chatIn} out=${chatOut} total=${chatTotal} ` +
        `(acc_total=${TOKENS.chat_total}, calls=${TOKENS.chat_calls})`
    );
  }

  // 응답 형태 호환 처리
  return {
    content:
      json?.result?.message?.content?.[0]?.text || // HCX-005 구조
      json?.result?.message?.content ||
      "",
    tokens: {
      input: chatIn,
      output: chatOut,
      total: chatTotal,
    },
  };
}

// [수정] 기존 appendToLogSheet 함수를 아래 코드로 전체 교체합니다.
async function appendToLogSheet(socketId, messagesToLog) {
  const credentials = {
    LOG_GOOGLE_SHEET_ID,
    LOG_GOOGLE_SHEET_NAME,
    GOOGLE_SERVICE_ACCOUNT_EMAIL,
    GOOGLE_PRIVATE_KEY,
  };
  if (Object.values(credentials).some((v) => !v)) {
    console.warn("[Google Sheets Log] Credentials not set. Skipping log.");
    return;
  }

  try {
    const auth = new google.auth.GoogleAuth({
      /* ... */
    });
    const sheets = google.sheets({ version: "v4", auth });

    const logData = chatLogs.get(socketId);
    const history = chatHistories.get(socketId) || [];

    // 이전에 기록한 row가 있는지 확인
    if (logData && logData.rowNumber) {
      // --- 업데이트 로직 ---
      // 이전에 기록된 메시지 수를 기반으로 시작 열을 계산 (C열이 2)
      // history.length는 현재 메시지까지 포함하므로 2를 빼서 이전 길이를 구함
      const startColumnIndex = 2 + (history.length - messagesToLog.length);
      const endColumnIndex = startColumnIndex + messagesToLog.length - 1;

      const startColumn = columnIndexToLetter(startColumnIndex);
      const endColumn = columnIndexToLetter(endColumnIndex);

      const range = `${LOG_GOOGLE_SHEET_NAME}!${startColumn}${logData.rowNumber}:${endColumn}${logData.rowNumber}`;

      await sheets.spreadsheets.values.update({
        spreadsheetId: LOG_GOOGLE_SHEET_ID,
        range: range,
        valueInputOption: "USER_ENTERED",
        resource: { values: [messagesToLog] },
      });
      console.log(`[Google Sheets Log] Updated row ${logData.rowNumber}`);
    } else {
      // --- 첫 기록 (Append) 로직 ---
      const timestamp = new Date()
        .toLocaleDateString("ko-KR", {
          /* ... */
        })
        .replace(/ /g, "");
      const systemPrompt = logData ? logData.systemPrompt : defaultSystemPrompt;
      const rowData = [timestamp, systemPrompt, ...messagesToLog];

      const response = await sheets.spreadsheets.values.append({
        spreadsheetId: LOG_GOOGLE_SHEET_ID,
        range: LOG_GOOGLE_SHEET_NAME,
        valueInputOption: "USER_ENTERED",
        resource: { values: [rowData] },
      });

      // 응답에서 새로 추가된 row의 번호를 추출
      const updatedRange = response.data.updates.updatedRange;
      const match = updatedRange.match(/!A(\d+):/);
      if (match && match[1] && logData) {
        logData.rowNumber = parseInt(match[1], 10);
        console.log(
          `[Google Sheets Log] Appended new row ${logData.rowNumber}`
        );
      }
    }
  } catch (error) {
    console.error("Error logging to Google Sheet:", error.message);
  }
}
// // [추가] 채팅 로그를 Google Sheet에 추가하는 함수
// async function appendToLogSheet(rowData) {
//   // 로그 시트 정보가 .env에 없으면 함수를 조용히 종료
//   if (
//     !LOG_GOOGLE_SHEET_ID ||
//     !LOG_GOOGLE_SHEET_NAME ||
//     !GOOGLE_SERVICE_ACCOUNT_EMAIL ||
//     !GOOGLE_PRIVATE_KEY
//   ) {
//     console.warn(
//       "[Google Sheets Log] Logging credentials not set in .env. Skipping log append."
//     );
//     return;
//   }

//   try {
//     const auth = new google.auth.GoogleAuth({
//       credentials: {
//         client_email: GOOGLE_SERVICE_ACCOUNT_EMAIL,
//         private_key: GOOGLE_PRIVATE_KEY,
//       },
//       // 읽기/쓰기 권한이 필요
//       scopes: ["https://www.googleapis.com/auth/spreadsheets"],
//     });

//     const sheets = google.sheets({ version: "v4", auth });
//     await sheets.spreadsheets.values.append({
//       spreadsheetId: LOG_GOOGLE_SHEET_ID,
//       range: LOG_GOOGLE_SHEET_NAME, // 시트 이름! A1 표기법이 아님
//       valueInputOption: "USER_ENTERED",
//       resource: {
//         values: [rowData], // rowData는 배열이어야 함. 예: ['val1', 'val2']
//       },
//     });
//     console.log(`[Google Sheets Log] Successfully appended a row.`);
//   } catch (error) {
//     console.error("Error appending to Google Sheet:", error.message);
//   }
// }

// [추가] Google Sheets 데이터 로더 함수
async function loadDataFromGoogleSheet() {
  if (
    !GOOGLE_SHEET_ID ||
    !GOOGLE_SHEET_RANGE ||
    !GOOGLE_SERVICE_ACCOUNT_EMAIL ||
    !GOOGLE_PRIVATE_KEY
  ) {
    throw new Error("Google Sheets API credentials are not set in .env file.");
  }

  const auth = new google.auth.GoogleAuth({
    credentials: {
      client_email: GOOGLE_SERVICE_ACCOUNT_EMAIL,
      private_key: GOOGLE_PRIVATE_KEY,
    },
    scopes: ["https://www.googleapis.com/auth/spreadsheets.readonly"],
  });

  const sheets = google.sheets({ version: "v4", auth });
  const response = await sheets.spreadsheets.values.get({
    spreadsheetId: GOOGLE_SHEET_ID,
    range: GOOGLE_SHEET_RANGE,
  });

  const rows = response.data.values;
  if (!rows || rows.length === 0) {
    console.log("No data found in Google Sheet.");
    return [];
  }

  // 첫 번째 행을 헤더(key)로 사용
  const headers = rows[0].map((h) => String(h || "").trim());
  // 나머지 행들을 { header: value } 형태의 객체 배열로 변환
  const data = rows.slice(1).map((row) => {
    const rowData = {};
    headers.forEach((header, index) => {
      rowData[header] = row[index];
    });
    return rowData;
  });

  console.log(`[Google Sheets] Loaded ${data.length} rows.`);
  return data;
}

// 컬럼 별칭
const FIELD_ALIASES = {
  category: ["행사분류", "행사구분"],
  industry: [
    "행사분야",
    "산업군",
    "산업분야",
    "카테고리",
    "분야",
    "industry",
    "Category",
  ],
  title: ["행사명"],
  subtitle: ["행사명(서브타이틀)"],
  date: ["행사 시작일자", "날짜", "기간", "개최기간", "전시기간", "date"],
  endDate: ["행사 종료일자"],
  venue: ["행사 장소", "장소", "전시장", "개최장소", "Hall", "venue"],
  price: ["입장료"],
  host: ["주최"],
  manage: ["주관"],
  inquiry: ["담당자/공연문의 정보", "단체문의 정보", "예매문의 정보"],
  site: ["관련 사이트"],
  ticket: ["티켓 예약"],
  age: ["추천 연령대", "연령대", "나이"],
  gender: ["성별"],
  interest: ["관심사"],
  job: ["직업"],
};

function pickByAliases(row, aliases) {
  // 1) 정확 매칭
  for (const k of aliases) {
    const v = row[k];
    if (v && String(v).trim()) return String(v).trim();
  }
  // 2) 느슨 매칭(공백 제거 후 포함 관계)
  const keys = Object.keys(row);
  for (const k of keys) {
    const nk = k.replace(/\s+/g, "");
    const hit = aliases.find((a) => nk.includes(String(a).replace(/\s+/g, "")));
    if (hit) {
      const v = row[k];
      if (v && String(v).trim()) return String(v).trim();
    }
  }
  return "";
}

function mapRow(r) {
  // FIELD_ALIASES에 정의된 모든 키에 대해 값을 추출합니다.
  const category = pickByAliases(r, FIELD_ALIASES.category);
  const industry = pickByAliases(r, FIELD_ALIASES.industry);
  const title = pickByAliases(r, FIELD_ALIASES.title);
  const subtitle = pickByAliases(r, FIELD_ALIASES.subtitle);
  const startDate = pickByAliases(r, FIELD_ALIASES.date);
  const endDate = pickByAliases(r, FIELD_ALIASES.endDate);
  const venue = pickByAliases(r, FIELD_ALIASES.venue);
  const price = pickByAliases(r, FIELD_ALIASES.price);
  const host = pickByAliases(r, FIELD_ALIASES.host);
  const manage = pickByAliases(r, FIELD_ALIASES.manage);
  const inquiry = pickByAliases(r, FIELD_ALIASES.inquiry);
  const site = pickByAliases(r, FIELD_ALIASES.site);
  const ticket = pickByAliases(r, FIELD_ALIASES.ticket);
  const age = pickByAliases(r, FIELD_ALIASES.age);
  const gender = pickByAliases(r, FIELD_ALIASES.gender);
  const interest = pickByAliases(r, FIELD_ALIASES.interest);
  const job = pickByAliases(r, FIELD_ALIASES.job);

  // 시작일과 종료일을 하나의 날짜 문자열로 조합합니다.
  let date = startDate;
  if (startDate && endDate && startDate !== endDate) {
    date = `${startDate} ~ ${endDate}`;
  }

  // 제목과 부제를 합칩니다.
  const fullTitle = subtitle ? `${title} (${subtitle})` : title;

  // 추출한 모든 정보를 조합하여 임베딩에 사용할 baseText를 만듭니다.
  // 각 정보 앞에 태그를 붙여주면 의미를 더 명확하게 할 수 있습니다.
  const baseText = [
    fullTitle,
    category && `분류/구분:${category}`,
    industry && `행사분야:${industry}`,
    date && `기간:${date}`,
    venue && `장소:${venue}`,
    price && `입장료:${price}`,
    age && `추천연령:${age}`,
    gender && `성별:${gender}`,
    interest && `관심사:${interest}`,
    job && `직업:${job}`,
    host && `주최:${host}`,
    manage && `주관:${manage}`,
    inquiry && `문의:${inquiry}`,
    site && `웹사이트:${site}`,
    ticket && `티켓 예약:${ticket}`,
  ]
    .filter(Boolean) // 내용이 없는 항목은 제외합니다.
    .join(" / ");

  // 나중에 활용할 수 있도록 모든 필드를 객체로 반환합니다.
  return {
    category,
    industry,
    title,
    subtitle,
    date,
    venue,
    price,
    host,
    manage,
    inquiry,
    site,
    ticket,
    age,
    gender,
    interest,
    job,
    baseText,
  };
}

async function buildVectors() {
  console.log("Fetching data from Google Sheets...");
  const rows = await loadDataFromGoogleSheet(); // Google Sheets에서 데이터 가져오기
  const out = [];
  // const rows = await loadCsvRows(DATA_CSV);
  // const out = [];
  console.log(`Starting to build vectors for ${rows.length} rows...`);

  for (let i = 0; i < rows.length; i++) {
    try {
      const m = mapRow(rows[i]);
      if (!m.baseText || m.baseText.length < 2) continue;
      const segments =
        m.baseText.length > 2000 ? await segmentText(m.baseText) : [m.baseText];
      for (const seg of segments) {
        if (!seg || !seg.trim()) continue;
        const embedding = await embedText(seg);
        out.push({ id: `${i}-${out.length}`, meta: m, text: seg, embedding });
      }
      // API 속도 제한을 피하기 위한 짧은 대기
      await new Promise((r) => setTimeout(r, 1000));
    } catch (e) {
      console.error(`[row ${i}]`, e.message);
    }
  }

  if (!out.length)
    throw new Error("No embeddings produced from Google Sheet data.");
  const tmp = VECTORS_JSON + ".tmp";
  fs.writeFileSync(tmp, JSON.stringify(out, null, 2), "utf8");
  fs.renameSync(tmp, VECTORS_JSON);

  console.log(`Successfully built ${out.length} vectors.`);
  return out.length;
}

function removeEmojiLikeExpressions(text) {
  if (typeof text !== "string") return ""; // 입력값이 문자열이 아닌 경우 빈 문자열 반환
  return (
    text
      // 이모지 제거 (유니코드 속성 사용)
      .replace(/[\p{Emoji_Presentation}\p{Extended_Pictographic}]/gu, "")
      // ㅎㅎ, ㅋㅋ, ㅠㅠ, ^^, ^^;;, ;; 등 반복 감정 표현 제거
      .replace(/([ㅎㅋㅠ]+|[\^]+|;{2,})/g, "")
      // 여러 번 반복된 공백을 하나로 정리
      .replace(/\s{2,}/g, " ")
      .trim()
  );
}

// ========== (1) 전처리/임베딩 (보강판) ==========
app.post("/pre_processing_for_embedding", async (_req, res) => {
  try {
    // if (!fs.existsSync(DATA_CSV)) {
    //   return res
    //     .status(400)
    //     .json({ ok: false, error: `CSV not found at ${DATA_CSV}` });
    // }
    const count = await buildVectors();
    res.json({ ok: true, count, file: "data/vectors.json" });
    logTokenSummary("after build");
  } catch (e) {
    console.error(e); // [추가] 에러 로그
    res.status(500).json({ ok: false, error: String(e) });
  }
});

// ========== (2) 질의/시맨틱검색/프롬프트+생성 ==========
app.post("/query_with_embedding", async (req, res) => {
  try {
    const question = (req.body?.question || "").trim();
    if (!question) return res.status(400).json({ error: "question required" });
    if (!fs.existsSync(VECTORS_JSON)) {
      return res.status(400).json({
        error:
          "vectors.json not found. Run /pre_processing_for_embedding first.",
      });
    }
    // 🟢 새 systemPrompt 처리
    const activeSystemPrompt =
      (req.body?.systemPrompt && req.body.systemPrompt.trim()) ||
      defaultSystemPrompt;

    const cid = getCid(req);
    const prev = chatHistories.get(cid) || [];

    const vectors = JSON.parse(fs.readFileSync(VECTORS_JSON, "utf8"));
    if (!Array.isArray(vectors) || vectors.length === 0) {
      return res.status(400).json({
        error: "vectors.json is empty. Re-run /pre_processing_for_embedding.",
      });
    }

    const qEmb = await embedText(question);

    const scored = vectors
      .map((v) => ({ v, score: cosineSim(qEmb, v.embedding) }))
      .sort((a, b) => b.score - a.score);

    const ranked = scored.slice(0, TOP_K);
    const slimHits = ranked.map(({ v, score }) => ({
      id: v.id,
      meta: v.meta,
      text: v.text,
      score: Number(score.toFixed(4)),
    }));

    const context = slimHits
      .map((h, i) => {
        const m = h.meta || {};
        return (
          `[${i + 1}] ${m.title || ""} | ${m.date || ""} | ${m.venue || ""}` +
          `${m.region ? " | 지역:" + m.region : ""}` +
          `${m.industry ? " | 산업군:" + m.industry : ""}\n` +
          h.text
        );
      })
      .join("\n\n");

    const messages = [
      {
        role: "system",
        content: activeSystemPrompt,
      },
      ...prev, // ← 추가 (대화 맥락)
      {
        role: "user",
        content: `질문: ${question}\n\n[참고 가능한 이벤트]\n${context}\n\n위 정보만 사용해 사용자 질문에 답하세요.`,
      },
    ];

    const result = await callClovaChat(messages, {
      temperature: 0.3,
      maxTokens: 700,
    });
    // 🔹 히스토리 업데이트 (유저 질문 / 모델 응답)
    pushHistory(cid, "user", question);
    pushHistory(cid, "assistant", result.content);

    res.json({
      answer: result.content,
      hits: slimHits,
      conversationId: cid,
      tokens: result.tokens,
    });
    logTokenSummary("after query");
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e) });
  }
});

app.post("/reset_conversation", (req, res) => {
  const cid = getCid(req);
  chatHistories.delete(cid);
  res.json({ ok: true, conversationId: cid });
});

// (옵션) 헬스체크
app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    appId: APP_ID,
    embedBase: HLX_BASE,
    chatBase: CLOVA_BASE,
    embedModel: EMB_MODEL,
    topK: TOP_K,
  });
});

// ==================== [수정 코드 시작] ====================
// 숫자를 Excel 열 문자로 변환하는 헬퍼 함수 (예: 0 -> A, 1 -> B, 26 -> AA)
function columnIndexToLetter(index) {
  let temp,
    letter = "";
  while (index >= 0) {
    temp = index % 26;
    letter = String.fromCharCode(temp + 65) + letter;
    index = Math.floor(index / 26) - 1;
  }
  return letter;
}
// ==================== [수정 코드 종료] ====================

io.on("connection", (socket) => {
  console.log(`✅ [socket] connected: ${socket.id}`);
  chatHistories.set(socket.id, []);
  // [수정] 로그 데이터에 rowNumber를 추가하여 초기화
  chatLogs.set(socket.id, {
    systemPrompt: defaultSystemPrompt,
    startTime: new Date(),
    rowNumber: null, // 아직 몇 번째 줄에 기록될지 모름
  });

  // 'go' 버튼 클릭 시 대화 시작 (인사말 생성)
  socket.on("start-conversation", async (payload) => {
    try {
      const activeSystemPrompt =
        (payload.systemPrompt && payload.systemPrompt.trim()) ||
        defaultSystemPrompt;

      // 시스템 프롬프트에 정의된 인사말을 생성하기 위해 초기 메시지 구성
      const messages = [
        { role: "system", content: activeSystemPrompt },
        // 모델이 첫 응답(인사말)을 생성하도록 유도하는 메시지
        { role: "user", content: "안녕하세요." },
      ];

      // LLM 호출하여 인사말 생성
      const result = await callClovaChat(messages, {
        temperature: 0.5, // 인사말이므로 약간의 창의성을 허용
        maxTokens: 300,
      });

      const cleanedAnswer = removeEmojiLikeExpressions(result.content);

      // 구글 시트 로깅 추가
      const timestamp = new Date()
        .toLocaleDateString("ko-KR", {
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
        })
        .replace(/ /g, "");

      // 첫 로그에는 시스템 프롬프트와 생성된 인사말을 기록합니다.
      // 사용자 질문은 "안녕하세요."로 고정됩니다.
      appendToLogSheet([
        timestamp,
        activeSystemPrompt,
        "안녕하세요.",
        // result.content,
        cleanedAnswer,
      ]);

      // 생성된 인사말을 대화 기록에 추가
      // (사용자가 입력한 "안녕하세요."는 실제 입력이 아니므로 기록하지 않음)
      pushHistory(socket.id, "assistant", cleanedAnswer);

      // 클라이언트로 인사말 응답 전송
      socket.emit("reply", {
        answer: cleanedAnswer,
        hits: [], // 첫 인사에는 참조 정보가 없음
        tokens: result.tokens,
      });

      logTokenSummary("after start-conversation");
    } catch (e) {
      console.error("[socket start-conversation error]", e);
      socket.emit("reply", { error: String(e) });
    }
  });

  socket.on("message", async (payload) => {
    try {
      // const q = String(question || "").trim();
      const q =
        typeof payload === "string"
          ? payload
          : String(payload.question || "").trim();
      if (!q) return socket.emit("reply", { error: "question required" });

      if (!fs.existsSync(VECTORS_JSON)) {
        return socket.emit("reply", {
          error:
            "vectors.json not found. Run /pre_processing_for_embedding first.",
        });
      }

      const activeSystemPrompt =
        (payload.systemPrompt && payload.systemPrompt.trim()) ||
        defaultSystemPrompt;

      const vectors = JSON.parse(fs.readFileSync(VECTORS_JSON, "utf8"));
      if (!Array.isArray(vectors) || vectors.length === 0) {
        return socket.emit("reply", {
          error: "vectors.json is empty. Re-run /pre_processing_for_embedding.",
        });
      }

      // 1) 질문 임베딩
      const qEmb = await embedText(q);

      // 2) 로컬 시맨틱 검색
      const scored = vectors
        .map((v) => ({ v, score: cosineSim(qEmb, v.embedding) }))
        .sort((a, b) => b.score - a.score);

      const ranked = scored.slice(0, TOP_K);
      const slimHits = ranked.map(({ v, score }) => ({
        id: v.id,
        meta: v.meta,
        text: v.text,
        score: Number(score.toFixed(4)),
      }));

      // 3) 컨텍스트 문자열
      const context = slimHits
        .map((h, i) => {
          const m = h.meta || {};
          return (
            `[${i + 1}] ${m.title || ""} | ${m.date || ""} | ${m.venue || ""}` +
            `${m.region ? " | 지역:" + m.region : ""}` +
            `${m.industry ? " | 산업군:" + m.industry : ""}\n` +
            h.text
          );
        })
        .join("\n\n");

      // 4) 이전 히스토리 포함 메시지 구성
      const prev = chatHistories.get(socket.id) || [];
      const messages = [
        { role: "system", content: activeSystemPrompt },
        ...prev,
        {
          role: "user",
          content: `질문: ${q}\n\n[참고 가능한 이벤트]\n${context}\n\n위 정보만 사용해 사용자 질문에 답하세요.`,
        },
      ];

      // 5) LLM 호출
      const result = await callClovaChat(messages, {
        temperature: 0.3,
        maxTokens: 700,
      });

      const cleanedAnswer = removeEmojiLikeExpressions(result.content);

      // 6) 구글 시트 로깅
      const isFirstMessage = prev.length === 0;
      if (isFirstMessage) {
        // "YYYY.MM.DD." 형식으로 날짜 생성
        const timestamp = new Date()
          .toLocaleDateString("ko-KR", {
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
          })
          .replace(/ /g, "");
        appendToLogSheet([timestamp, activeSystemPrompt, q, cleanedAnswer]);
      } else {
        appendToLogSheet(["", "", q, cleanedAnswer]);
      }

      // 7) 히스토리 업데이트
      pushHistory(socket.id, "user", q);
      pushHistory(socket.id, "assistant", cleanedAnswer);

      // 8) 응답 전송
      socket.emit("reply", {
        answer: cleanedAnswer,
        hits: slimHits,
        tokens: result.tokens,
      });
      logTokenSummary("after ws query");
    } catch (e) {
      console.error("[socket message error]", e);
      socket.emit("reply", { error: String(e) });
    }
  });

  // 대화 리셋
  socket.on("reset", () => {
    chatHistories.set(socket.id, []);
    socket.emit("reply", { ok: true, reset: true });
  });

  socket.on("disconnect", () => {
    chatHistories.delete(socket.id);
    console.log(`❎ [socket] disconnected: ${socket.id}`);
  });
});

// 서버 시작 시 `vectors.json` 파일이 없으면 Google Sheets 기반으로 생성
async function buildVectorsIfMissing() {
  const vectorsExist = fs.existsSync(VECTORS_JSON);
  let needBuild = !vectorsExist;

  if (vectorsExist) {
    try {
      const arr = JSON.parse(fs.readFileSync(VECTORS_JSON, "utf8"));
      if (!Array.isArray(arr) || arr.length === 0) {
        needBuild = true; // 파일은 있지만 내용이 비어있으면 재생성
      }
    } catch {
      needBuild = true; // 파일이 깨져있으면 재생성
    }
  }

  if (needBuild) {
    console.log(
      "🔧 vectors.json missing or empty → building from Google Sheets..."
    );
    const count = await buildVectors();
    console.log(`✅ vectors.json ready: ${count} items`);
  } else {
    console.log(
      "✅ vectors.json already exists. Use the API to rebuild if needed."
    );
  }
}

// 서버 시작부
const PORT = process.env.PORT || 3000;
buildVectorsIfMissing()
  // ensureVectorsUpToDate()
  .catch((err) => console.error("[warmup error]", err))
  .finally(() => {
    server.listen(PORT, () => console.log(`http://localhost:${PORT}`));
  });
