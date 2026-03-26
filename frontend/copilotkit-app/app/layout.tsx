import type { Metadata } from "next";
import "@copilotkit/react-ui/styles.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "POP Agent CopilotKit UI",
  description: "Web frontend for the POP agent runtime using CopilotKit and a trusted component contract.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
