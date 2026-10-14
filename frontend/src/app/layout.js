import "./globals.css";

export const metadata = {
  title: "ResoGenre — Music Genre Classification",
  description: "AI-powered music genre classification using deep learning. Upload any audio file and instantly identify its genre with our 4-stage CNN model.",
  keywords: "music genre classification, deep learning, CNN, audio analysis, machine learning",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>
        {children}
      </body>
    </html>
  );
}
