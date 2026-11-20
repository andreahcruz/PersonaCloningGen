import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import type { ReactNode } from "react";
import AppShell from "@/components/AppShell";
import { PREFS_INIT_SCRIPT } from "@/lib/prefs";
import "./globals.css";

const plex = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-plex",
  display: "swap",
});
const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Lemkin Studio",
  description: "AI-powered content generation inspired by Jason Lemkin's writing style",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    // data-theme and data-persona are set by the inline script before paint, so the server markup
    // and the browser can differ on purpose.
    <html
      lang="en"
      className={`${plex.variable} ${plexMono.variable} antialiased`}
      data-theme="light"
      data-persona="lemkin"
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: PREFS_INIT_SCRIPT }} />
      </head>
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
