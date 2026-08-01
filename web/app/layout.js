import "./globals.css";

export const metadata = {
  title: "Cognitive State Detection",
  description:
    "Privacy-preserving engagement detection running entirely in your browser",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
