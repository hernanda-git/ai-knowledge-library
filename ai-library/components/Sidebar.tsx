"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

interface SidebarProps {
  title: string;
  links: { label: string; href: string }[];
}

export default function Sidebar({ title, links }: SidebarProps) {
  const path = usePathname();
  return (
    <aside className="sb">
      <div className="sc">{title}</div>
      <Link href="/">🏠 Home</Link>
      <Link href={path.split("/").slice(0,2).join("/")} className="active">📂 Overview</Link>
      <div className="sc">Sections</div>
      {links.map(l => <Link key={l.href} href={l.href}>{l.label}</Link>)}
    </aside>
  );
}
