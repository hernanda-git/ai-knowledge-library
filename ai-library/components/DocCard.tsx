import { KnowledgeDoc } from "@/lib/types";

export default function DocCard({ doc, icon }: { doc: KnowledgeDoc; icon?: string }) {
  return (
    <div className="card" id={doc.id}>
      <h3>{icon || "📄"} {doc.title} <span style={{ fontSize: "11px", color: "var(--tx3)", fontWeight: 400 }}>— {doc.lines} lines</span></h3>
      {doc.description && <p>{doc.description}</p>}
      {doc.tags.length > 0 && (
        <div style={{ margin: "8px 0", display: "flex", flexWrap: "wrap", gap: "4px" }}>
          {doc.tags.map(t => <span key={t} className="tag ac">{t}</span>)}
        </div>
      )}
      {doc.sections.map(s => (
        <div key={s.title}>
          <h4>{s.title}</h4>
          <div dangerouslySetInnerHTML={{ __html: s.content }} />
        </div>
      ))}
    </div>
  );
}
