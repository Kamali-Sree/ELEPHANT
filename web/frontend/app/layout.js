import "./globals.css";
import Link from "next/link";

export const metadata = {
  title: "MACONFLIC — Elephant Conflict Monitoring Platform",
  description: "AI-Based Elephant Vocalization Detection & Human-Wildlife Conflict Monitoring System. Detect, classify, and monitor elephant vocalizations using deep learning.",
  keywords: "elephant, vocalization, AI, deep learning, wildlife, conflict, monitoring, GIS",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet" />
      </head>
      <body>
        <Navigation />
        <main>{children}</main>
      </body>
    </html>
  );
}

function Navigation() {
  const links = [
    { href: "/", label: "Dashboard", icon: "📊" },
    { href: "/map", label: "GIS Map", icon: "🗺️" },
    { href: "/analyze", label: "Analyzer", icon: "🔬" },
    { href: "/report", label: "Report", icon: "📝" },
    { href: "/alerts", label: "Alerts", icon: "🔔" },
  ];

  return (
    <nav className="nav-container">
      <div className="nav-inner">
        <Link href="/" className="nav-logo">
          🐘 <span>MACONFLIC</span>
        </Link>
        <div className="nav-links">
          {links.map((link) => (
            <Link key={link.href} href={link.href} className="nav-link">
              {link.icon} {link.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
