"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { categories } from "@/lib/content";

export default function Navbar() {
  const path = usePathname();
  return (
    <nav className="nav">
      <div className="brand"><span>📖 AI Library</span><span className="badge">46 Docs</span></div>
      <Link href="/" className={path === "/" ? "active" : ""}>Home</Link>
      {categories.map(c => (
        <Link key={c.id} href={`/${c.id}`} className={path.startsWith(`/${c.id}`) ? "active" : ""}>
          {c.icon} {c.title}
        </Link>
      ))}
    </nav>
  );
}
