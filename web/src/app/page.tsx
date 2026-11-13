"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  FiSend, FiSettings, FiInfo, FiX, FiUploadCloud, FiTrash2, FiFileText,
  FiGitPullRequest, FiMenu,
} from "react-icons/fi";

const DEFAULT_API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Citation = { index: number; source: string; page: number | null; snippet: string };
type Doc = { id: string; name: string };
type Msg = { role: "user" | "assistant"; text: string; citations?: Citation[] };

export default function Home() {
  const [apiBase, setApiBase] = useState(DEFAULT_API);
  const [showSettings, setShowSettings] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [docs, setDocs] = useState<Doc[]>([]);
  const [docType, setDocType] = useState("other");
  const [uploading, setUploading] = useState(false);

  const [messages, setMessages] = useState<Msg[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hybrid, setHybrid] = useState(true);
  const [rerank, setRerank] = useState(true);

  const [cmpA, setCmpA] = useState("");
  const [cmpB, setCmpB] = useState("");
  const [cmpTopic, setCmpTopic] = useState("");
  const [comparing, setComparing] = useState(false);

  const fileRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  const base = () => apiBase.replace(/\/$/, "");

  useEffect(() => {
    try {
      const s = localStorage.getItem("cv_api_base");
      if (s) setApiBase(s);
    } catch {}
  }, []);
  useEffect(() => { refreshDocs(); /* eslint-disable-next-line */ }, [apiBase]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);

  function saveApi(v: string) {
    setApiBase(v);
    try { localStorage.setItem("cv_api_base", v); } catch {}
  }

  async function refreshDocs() {
    try {
      const r = await fetch(`${base()}/documents`);
      if (r.ok) setDocs(await r.json());
    } catch {}
  }

  async function upload() {
    const files = fileRef.current?.files;
    if (!files || files.length === 0) return;
    setUploading(true); setError("");
    try {
      const fd = new FormData();
      Array.from(files).forEach((f) => fd.append("files", f));
      fd.append("document_type", docType);
      const r = await fetch(`${base()}/documents`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`Upload failed (${r.status})`);
      if (fileRef.current) fileRef.current.value = "";
      await refreshDocs();
    } catch (e) {
      setError(`Upload failed. Is the backend reachable at ${apiBase}? ${e instanceof Error ? e.message : ""}`);
    } finally { setUploading(false); }
  }

  async function removeDoc(id: string) {
    try {
      await fetch(`${base()}/documents/${id}`, { method: "DELETE" });
      await refreshDocs();
    } catch {}
  }

  async function ask(q?: string) {
    const text = (q ?? question).trim();
    if (!text || loading) return;
    setMessages((m) => [...m, { role: "user", text }]);
    setQuestion(""); setLoading(true); setError("");
    try {
      const r = await fetch(`${base()}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session: "web", question: text, hybrid, rerank }),
      });
      if (r.status === 503) {
        setMessages((m) => [...m, { role: "assistant", text: "No documents are indexed yet. Upload a document on the left, then ask again." }]);
        return;
      }
      if (!r.ok) throw new Error(`API ${r.status}`);
      const d = await r.json();
      setMessages((m) => [...m, { role: "assistant", text: d.answer, citations: d.citations }]);
    } catch (e) {
      setError(`Could not reach the backend at ${apiBase}. It may be waking up (free tier sleeps — first call ~50s). ${e instanceof Error ? e.message : ""}`);
    } finally { setLoading(false); }
  }

  async function compare() {
    if (!cmpA || !cmpB || cmpA === cmpB || comparing) return;
    setComparing(true); setError("");
    const a = docs.find((d) => d.id === cmpA)?.name || "A";
    const b = docs.find((d) => d.id === cmpB)?.name || "B";
    setMessages((m) => [...m, { role: "user", text: `Compare "${a}" vs "${b}"${cmpTopic ? ` — ${cmpTopic}` : ""}` }]);
    try {
      const r = await fetch(`${base()}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id_a: cmpA, document_id_b: cmpB, topic: cmpTopic || null }),
      });
      if (!r.ok) throw new Error(`API ${r.status}`);
      const d = await r.json();
      setMessages((m) => [...m, { role: "assistant", text: d.report }]);
    } catch (e) {
      setError(`Comparison failed. ${e instanceof Error ? e.message : ""}`);
    } finally { setComparing(false); }
  }

  return (
    <div className="flex h-[100dvh] overflow-hidden">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0 fixed md:static z-30 w-72 h-[100dvh] bg-black/40 backdrop-blur border-r border-white/10 flex flex-col transition-transform`}>
        <div className="p-4 overflow-y-auto flex-1 space-y-5">
          {/* Upload */}
          <div>
            <p className="text-[0.65rem] uppercase tracking-widest text-neutral-400 mb-2">Add documents</p>
            <input ref={fileRef} type="file" multiple accept=".pdf,.txt,.md"
              className="text-xs text-neutral-300 w-full mb-2 file:mr-2 file:rounded-md file:border-0 file:bg-violet-600 file:text-white file:px-2 file:py-1 file:text-xs" />
            <select value={docType} onChange={(e) => setDocType(e.target.value)} className="field mb-2 text-xs">
              <option value="other">General</option>
              <option value="resume">Policy / HR</option>
              <option value="job_description">Contract / SLA</option>
            </select>
            <button onClick={upload} disabled={uploading} className="btn w-full">
              <FiUploadCloud /> {uploading ? "Uploading…" : "Upload & index"}
            </button>
          </div>

          {/* Document list */}
          <div>
            <p className="text-[0.65rem] uppercase tracking-widest text-neutral-400 mb-2">Documents ({docs.length})</p>
            <div className="space-y-1">
              {docs.length === 0 && <p className="text-xs text-neutral-500">None yet.</p>}
              {docs.map((d) => (
                <div key={d.id} className="group flex items-center gap-2 text-xs text-neutral-300 card px-2 py-1.5">
                  <FiFileText className="shrink-0 text-violet-300" />
                  <span className="flex-1 truncate">{d.name}</span>
                  <button onClick={() => removeDoc(d.id)} className="opacity-0 group-hover:opacity-100 text-neutral-500 hover:text-red-400">
                    <FiTrash2 size={12} />
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Version compare */}
          <div>
            <p className="text-[0.65rem] uppercase tracking-widest text-neutral-400 mb-2 flex items-center gap-1">
              <FiGitPullRequest /> Compare versions
            </p>
            <select value={cmpA} onChange={(e) => setCmpA(e.target.value)} className="field mb-1.5 text-xs">
              <option value="">Version A (older)…</option>
              {docs.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
            <select value={cmpB} onChange={(e) => setCmpB(e.target.value)} className="field mb-1.5 text-xs">
              <option value="">Version B (newer)…</option>
              {docs.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
            <input value={cmpTopic} onChange={(e) => setCmpTopic(e.target.value)} placeholder="Topic (optional)" className="field mb-2 text-xs" />
            <button onClick={compare} disabled={!cmpA || !cmpB || cmpA === cmpB || comparing} className="btn w-full">
              {comparing ? "Comparing…" : "Compare"}
            </button>
          </div>
        </div>
        <div className="p-3 border-t border-white/10 text-[0.65rem] text-neutral-500">
          Built by Suman Kumar ·{" "}
          <a className="text-violet-300" target="_blank" rel="noreferrer" href="https://github.com/sumankr0087/context-vault-rag">GitHub</a>
        </div>
      </aside>
      {sidebarOpen && <div className="fixed inset-0 bg-black/50 z-20 md:hidden" onClick={() => setSidebarOpen(false)} />}

      {/* Main */}
      <div className="flex-1 flex flex-col h-[100dvh] min-w-0">
        <header className="flex items-center justify-between px-4 py-3 border-b border-white/10">
          <div className="flex items-center gap-2 min-w-0">
            <button className="md:hidden text-neutral-300" onClick={() => setSidebarOpen(true)}><FiMenu size={20} /></button>
            <span className="text-xl">🔐</span>
            <div className="min-w-0">
              <h1 className="font-bold glow-text truncate leading-tight">ContextVault</h1>
              <p className="hidden sm:block text-[0.65rem] text-neutral-400 leading-tight">Ask your documents — grounded answers with citations</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={() => setShowAbout(true)} className="text-neutral-300 hover:text-violet-300 flex items-center gap-1 text-sm"><FiInfo size={16} /><span className="hidden sm:inline">About</span></button>
            <button onClick={() => setShowSettings((v) => !v)} className="text-neutral-400 hover:text-violet-300"><FiSettings size={18} /></button>
          </div>
        </header>

        {showSettings && (
          <div className="card m-3 p-3">
            <label className="text-xs uppercase tracking-wider text-neutral-400">Backend API URL</label>
            <input value={apiBase} onChange={(e) => saveApi(e.target.value)} className="field mt-1" />
          </div>
        )}

        {/* Conversation */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-3xl mx-auto space-y-4">
            {messages.length === 0 && !loading && (
              <div className="text-center mt-10">
                <h2 className="text-2xl font-bold glow-text mb-2">Ask your documents anything</h2>
                <p className="text-neutral-400 text-sm mb-6 max-w-xl mx-auto">
                  Upload policies, contracts or PDFs on the left, then ask questions. Answers are grounded in
                  your documents with clause-level citations and compliance notes.
                </p>
                <div className="flex flex-col gap-2 max-w-xl mx-auto">
                  {["Can I share client data externally?", "What is the travel reimbursement limit?", "What happens if SLA breaches occur?"].map((ex) => (
                    <button key={ex} onClick={() => ask(ex)} className="card px-4 py-3 text-sm text-left text-neutral-200 hover:border-violet-400/50 transition">{ex}</button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                {m.role === "user" ? (
                  <div className="bg-violet-600 text-white rounded-2xl rounded-br-sm px-4 py-2 text-sm max-w-[85%]">{m.text}</div>
                ) : (
                  <div className="card rounded-2xl rounded-bl-sm p-4 max-w-[92%] w-full">
                    <div className="prose-md text-sm text-neutral-200"><ReactMarkdown>{m.text}</ReactMarkdown></div>
                    {m.citations && m.citations.length > 0 && (
                      <div className="mt-3 border-t border-white/10 pt-2">
                        <p className="text-[0.65rem] uppercase tracking-widest text-neutral-500 mb-1">Citations</p>
                        <div className="space-y-1.5">
                          {m.citations.map((c) => (
                            <div key={c.index} className="text-xs text-neutral-400">
                              <span className="chip mr-1">[{c.index}]</span>
                              <span className="text-violet-300">{c.source}{c.page ? `, p.${c.page}` : ""}</span>
                              <span className="block text-neutral-500 mt-0.5">{c.snippet}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="card rounded-2xl px-4 py-3 text-sm text-neutral-400 animate-pulse">Searching documents & generating answer…</div>
              </div>
            )}
            {error && <div className="card p-4 text-sm text-amber-300 border-amber-400/30">{error}</div>}
            <div ref={endRef} />
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-white/10 px-4 py-3">
          <div className="max-w-3xl mx-auto">
            <div className="flex gap-2">
              <input value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask()}
                placeholder="Ask a question about your documents…" className="field rounded-full flex-1" />
              <button onClick={() => ask()} disabled={loading || !question.trim()} className="btn !rounded-full !px-4"><FiSend /></button>
            </div>
            <div className="flex items-center gap-4 mt-2 px-1 text-[0.7rem] text-neutral-400">
              <label className="flex items-center gap-1 cursor-pointer"><input type="checkbox" checked={hybrid} onChange={(e) => setHybrid(e.target.checked)} /> Hybrid search</label>
              <label className="flex items-center gap-1 cursor-pointer"><input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} /> Reranking</label>
            </div>
          </div>
        </div>
      </div>

      {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}
    </div>
  );
}

function AboutModal({ onClose }: { onClose: () => void }) {
  const items: [string, string][] = [
    ["📄 Grounded answers", "Every answer comes only from your uploaded documents — no made-up facts."],
    ["🔖 Clause-level citations", "Each claim cites the exact excerpt [n] with its source file and page."],
    ["🛡️ Hallucination guardrail", "If the documents don't cover it, it clearly says so instead of guessing."],
    ["⚠️ Compliance notes", "Flags risks, limits and obligations on policy/rule questions."],
    ["🔀 Version comparison", "Upload v1 & v2 of a policy and see exactly what changed."],
    ["🔎 Hybrid search + reranking", "Dense + keyword retrieval, then a cross-encoder reranks for accuracy."],
  ];
  return (
    <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4" onClick={onClose}>
      <div className="card max-w-lg w-full max-h-[85vh] overflow-y-auto p-6" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between mb-3">
          <h2 className="text-xl font-bold glow-text">About ContextVault</h2>
          <button onClick={onClose} className="text-neutral-400 hover:text-white"><FiX size={20} /></button>
        </div>
        <p className="text-sm text-neutral-300 leading-relaxed mb-4">
          <strong>ContextVault</strong> is an enterprise document & policy assistant. Upload your documents
          (PDF / TXT / Markdown), then ask questions in plain English — e.g. <em>"Can I share client data
          externally?"</em>. It retrieves the exact clauses, explains them simply, flags compliance risks,
          and can compare two versions of a policy.
        </p>
        <h3 className="font-semibold text-sm mb-2">What it does</h3>
        <ul className="space-y-2 mb-4 text-sm">
          {items.map(([t, d]) => (
            <li key={t}><span className="font-semibold text-violet-300">{t}</span><span className="text-neutral-400"> — {d}</span></li>
          ))}
        </ul>
        <p className="text-xs text-neutral-500">
          Built with LangChain · Chroma · hybrid search · cross-encoder reranking · Groq / Ollama. The demo
          backend is on a free tier that sleeps, so the first request may take ~50s. Uploaded docs are
          temporary on the free tier.{" "}
          <a className="text-violet-300" target="_blank" rel="noreferrer" href="https://github.com/sumankr0087/context-vault-rag">Source on GitHub</a>.
        </p>
      </div>
    </div>
  );
}
