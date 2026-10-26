import React from "react";
import "@fontsource/noto-sans-malayalam/400.css";
import "@fontsource/noto-sans-malayalam/700.css";
import "./globals.css";
export const metadata = {
  title: "Malayalam Voice AI",
  description: "Local scratch-trained Malayalam speech research baseline",
};
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
