import Link from "next/link";
import { categories, allDocs } from "@/lib/content";

export default function Home() {
  const totalLines = categories.reduce((a, c) => a + c.totalLines, 0);
  const totalDocs = categories.reduce((a, c) => a + c.docCount, 0);
  
  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "80px 32px 60px" }}>
      {/* Hero */}
      <div className="hero">
        <div className="t">AI Library of Alexandria</div>
        <h1>Complete AI Study — <span className="hl">{totalDocs} Documents</span></h1>
        <p>A unified, cross-referenced knowledge base covering the entire AI stack — from mathematical foundations and deep learning through LLMs, agents, RAG, enterprise deployment, multimodal AI, safety, interpretability, and the future roadmap. Auto-enriched every 12 hours.</p>
        <div className="stats">
          <div className="stat"><div className="n" style={{ color: "#818cf8" }}>{totalDocs}</div><div className="l">Documents</div></div>
          <div className="stat"><div className="n" style={{ color: "#22c55e" }}>{totalLines.toLocaleString()}</div><div className="l">Total Lines</div></div>
          <div className="stat"><div className="n" style={{ color: "#f59e0b" }}>{categories.length}</div><div className="l">Categories</div></div>
          <div className="stat"><div className="n" style={{ color: "#06b6d4" }}>46</div><div className="l">AI Topics</div></div>
        </div>
      </div>

      {/* Topic Categories */}
      <div className="g2">
        {categories.map(c => (
          <Link key={c.id} href={`/${c.id}`} style={{ textDecoration: "none" }} className="gc" >
            <div className="ic">{c.icon}</div>
            <h4>{c.title}</h4>
            <p>{c.docCount} docs · {c.totalLines.toLocaleString()} lines</p>
            <p style={{ fontSize: "10px", marginTop: 4, color: "var(--tx3)" }}>{c.description}</p>
          </Link>
        ))}
      </div>

      {/* Study Pathways */}
      <div className="card" style={{ marginTop: 32 }}>
        <h2>🧭 Study Pathways</h2>
        <p>Choose your learning path based on your role:</p>
        <div className="kp"><div className="ki" style={{ background: "rgba(99,102,241,0.15)" }}>🧠</div><div className="kt"><strong>ML Engineer:</strong> Math → ML Foundations → Deep Learning → Training Methodologies → LLM Models. <Link href="/foundations" style={{ color: "var(--accent-light)" }}>Start →</Link></div></div>
        <div className="kp"><div className="ki" style={{ background: "rgba(34,197,94,0.1)" }}>🔧</div><div className="kt"><strong>LLM Engineer:</strong> Transformer Architecture → Model Families → Tokenization → Quantization → Prompt Engineering → Evaluation. <Link href="/llms" style={{ color: "var(--accent-light)" }}>Start →</Link></div></div>
        <div className="kp"><div className="ki" style={{ background: "rgba(168,85,247,0.1)" }}>🤖</div><div className="kt"><strong>Agent Developer:</strong> Agent Architectures → MCP/ACP → Multi-Agent Systems → Agentic Frameworks → Tool Comparisons → SOUL/SKILL. <Link href="/agents" style={{ color: "var(--accent-light)" }}>Start →</Link></div></div>
        <div className="kp"><div className="ki" style={{ background: "rgba(6,182,212,0.1)" }}>📚</div><div className="kt"><strong>RAG Specialist:</strong> RAG Architectures → Advanced RAG → Vector Databases. <Link href="/rag" style={{ color: "var(--accent-light)" }}>Start →</Link></div></div>
        <div className="kp"><div className="ki" style={{ background: "rgba(245,158,11,0.1)" }}>🏭</div><div className="kt"><strong>Enterprise Architect:</strong> Enterprise Deployment → Fine-Tuning → AI Infrastructure → Evaluation. <Link href="/enterprise" style={{ color: "var(--accent-light)" }}>Start →</Link></div></div>
        <div className="kp"><div className="ki" style={{ background: "rgba(234,179,8,0.1)" }}>🚀</div><div className="kt"><strong>AI Researcher:</strong> Diffusion Models → Multimodal AI → Interpretability → Safety → Emerging Research → Papers. <Link href="/advanced" style={{ color: "var(--accent-light)" }}>Start →</Link></div></div>
        <div className="kp"><div className="ki" style={{ background: "rgba(239,68,68,0.1)" }}>🛡️</div><div className="kt"><strong>Safety Practitioner:</strong> AI Safety → Interpretability → Governance → Adversarial ML → Federated Learning. <Link href="/emerging" style={{ color: "var(--accent-light)" }}>Start →</Link></div></div>
      </div>

      {/* Stats */}
      <div className="card">
        <h2>📊 Library Statistics</h2>
        <div className="tw">
          <table>
            <thead><tr><th>Category</th><th>Docs</th><th>Lines</th><th>Key Topics</th></tr></thead>
            <tbody>
              {categories.map(c => (
                <tr key={c.id}>
                  <td>{c.icon} {c.title}</td>
                  <td>{c.docCount}</td>
                  <td>{c.totalLines.toLocaleString()}</td>
                  <td style={{ fontSize: "11px" }}>{c.description}</td>
                </tr>
              ))}
              <tr style={{ fontWeight: 700, color: "var(--tx1)" }}>
                <td><strong>Total</strong></td><td><strong>{totalDocs}</strong></td><td><strong>{totalLines.toLocaleString()}</strong></td><td><strong>2.1 MB</strong></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="ft">
        <p><strong>AI Library of Alexandria</strong> — Complete AI Study Reference</p>
        <p>C:\Workspace\AiBaseKnowledge\ · {totalDocs} Documents · {totalLines.toLocaleString()} Lines</p>
        <p style={{ marginTop: 4, fontSize: 11 }}>June 2026 · Auto-enriching every 12 hours</p>
      </div>
    </div>
  );
}
