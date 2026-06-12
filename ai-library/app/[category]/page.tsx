import { notFound } from "next/navigation";
import Link from "next/link";
import { categories, allDocs } from "@/lib/content";
import DocCard from "@/components/DocCard";
import Sidebar from "@/components/Sidebar";

export function generateStaticParams() {
  return categories.map(c => ({ category: c.id }));
}

export default async function TopicPage({ params }: { params: Promise<{ category: string }> }) {
  const { category } = await params;
  const cat = categories.find(c => c.id === category);
  if (!cat) notFound();
  
  const docs = allDocs[category] || [];

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      <Sidebar 
        title={`${cat.icon} ${cat.title}`}
        links={[
          { label: "📂 Overview", href: `/${cat.id}` },
          ...docs.map(d => ({ label: `📄 ${d.title}`, href: `#${d.id}` }))
        ]}
      />
      <main className="ct">
        <div className="hero">
          <div className="t">{cat.icon} {cat.title}</div>
          <h1>{cat.title} — <span className="hl">{docs.length} Documents</span></h1>
          <p>{cat.description}</p>
          <div className="stats">
            <div className="stat"><div className="n" style={{ color: cat.color }}>{docs.length}</div><div className="l">Documents</div></div>
            <div className="stat"><div className="n" style={{ color: "#22c55e" }}>{cat.totalLines.toLocaleString()}</div><div className="l">Total Lines</div></div>
          </div>
        </div>
        {docs.map(doc => (
          <DocCard key={doc.id} doc={doc} icon="📄" />
        ))}
        <div className="ft">
          <p><strong>AI Library of Alexandria</strong> — {cat.title} · {docs.length} Docs · {cat.totalLines.toLocaleString()} Lines</p>
          <p><Link href="/" style={{ color: "var(--accent-light)" }}>Home</Link> · June 2026</p>
        </div>
      </main>
    </div>
  );
}
