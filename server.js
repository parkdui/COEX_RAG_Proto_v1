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
app.use(express.static("public"));

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

// [추가] Google Sheets ENV 로드
const GOOGLE_SHEET_ID = getEnv("GOOGLE_SHEET_ID");
const GOOGLE_SHEET_RANGE = getEnv("GOOGLE_SHEET_RANGE");
const GOOGLE_SERVICE_ACCOUNT_EMAIL = getEnv("GOOGLE_SERVICE_ACCOUNT_EMAIL");
const GOOGLE_PRIVATE_KEY = getEnv("GOOGLE_PRIVATE_KEY");

// 파일 경로
// const DATA_CSV = path.join(__dirname, "data", "event_lists.csv");
const VECTORS_JSON = path.join(__dirname, "data", "vectors.json");
const systemPrompt = fs.readFileSync(
  path.join(__dirname, "LLM", "system_prompt.txt"),
  "utf8"
);

// --- systemPrompt 기본값 저장 (fallback) ---
let defaultSystemPrompt = systemPrompt;

// ---- Conversation Memory ----
const chatHistories = new Map();
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

// const { once } = require("events");
// 구분자 자동 감지 로더
// async function loadCsvRows(filePath) {
//   for (const sep of [",", ";", "\t"]) {
//     const rows = [];
//     const rs = fs.createReadStream(filePath).pipe(
//       csv({
//         separator: sep,
//         mapHeaders: ({ header }) => normalizeHeader(header),
//       })
//     );
//     rs.on("data", (r) => rows.push(r));
//     rs.on("error", (e) => console.error("[csv] stream error:", e));
//     await once(rs, "end");
//     // 키가 2개 이상 나오면 정상 파싱으로 판단
//     if (rows.length && Object.keys(rows[0]).length > 1) {
//       console.log(`[csv] parsed with separator "${sep}"`);
//       return rows;
//     }
//   }
//   // 마지막 시도라도 반환
//   const fallback = [];
//   const rs = fs.createReadStream(filePath).pipe(csv());
//   rs.on("data", (r) => fallback.push(r));
//   await once(rs, "end");
//   console.warn("[csv] fallback parser used");
//   return fallback;
// }

// function normalizeHeader(h) {
//   return String(h || "")
//     .replace(/^\uFEFF/, "")
//     .trim(); // BOM 제거
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
  // let baseText = [title, date, venue, region, industry, month && `월:${month}`]
  //   .filter(Boolean)
  //   .join(" / ");

  // // ⚠️ 아무것도 못 찾았으면: 해당 로우의 모든 값을 합쳐서라도 임베딩
  // if (!baseText || baseText.length < 2) {
  //   baseText = Object.values(r)
  //     .map((v) => String(v || "").trim())
  //     .filter(Boolean)
  //     .join(" / ");
  // }

  // return { title, date, venue, region, industry, month, baseText };
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
      await new Promise((r) => setTimeout(r, 250));
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

    const cid = getCid(req); // ← 추가
    const prev = chatHistories.get(cid) || []; // ← 추가

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
      meta: v.meta, // {title, date, venue, region, industry, month}
      text: v.text, // 검색에 사용된 원문
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

    // function wrapMessages(messages) {
    //   return messages.map((m) => ({
    //     role: m.role,
    //     content: [{ type: "text", text: m.content }],
    //   }));
    // }

    // const body = {
    //   messages: wrapMessages(messages),
    //   temperature: opts.temperature ?? 0.3,
    //   topP: opts.topP ?? 0.8,
    //   maxTokens: opts.maxTokens ?? 700,
    //   repetitionPenalty: 1.1, // 문서 기준 repeatPenalty → repetitionPenalty 이름도 확인
    //   stop: [],
    // };

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

io.on("connection", (socket) => {
  console.log(`✅ [socket] connected: ${socket.id}`);
  chatHistories.set(socket.id, []);

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

      // 6) 히스토리 업데이트
      pushHistory(socket.id, "user", q);
      pushHistory(socket.id, "assistant", result.content);

      // 7) 응답 전송
      socket.emit("reply", {
        answer: result.content,
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

// async function ensureVectorsUpToDate() {
//   if (!fs.existsSync(DATA_CSV)) {
//     console.warn("[warmup] CSV not found, skip warmup");
//     return;
//   }
//   const vecExists = fs.existsSync(VECTORS_JSON);
//   let need = !vecExists;

//   if (!need) {
//     try {
//       const csvM = fs.statSync(DATA_CSV).mtimeMs;
//       const vecM = fs.statSync(VECTORS_JSON).mtimeMs;
//       if (vecM < csvM) need = true;
//       const arr = JSON.parse(fs.readFileSync(VECTORS_JSON, "utf8"));
//       if (!Array.isArray(arr) || arr.length === 0) need = true;
//     } catch {
//       need = true;
//     }
//   }

//   if (need) {
//     console.log("🔧 vectors.json missing/stale → building...");
//     const count = await buildVectorsFromCsv();
//     console.log(`✅ vectors.json ready: ${count} items`);
//   } else {
//     console.log("✅ vectors.json up-to-date");
//   }
// }

// [수정] 서버 시작 시 `vectors.json` 파일이 없으면 Google Sheets 기반으로 생성
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
buildVectorsIfMissing() // [수정] 함수 이름 변경
  // ensureVectorsUpToDate()
  .catch((err) => console.error("[warmup error]", err))
  .finally(() => {
    server.listen(PORT, () => console.log(`http://localhost:${PORT}`));
  });
