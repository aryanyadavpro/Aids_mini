import "./globals.css";

export const metadata = {
  title: "ResoGenre — Music Genre Classification",
  description: "AI-powered music genre classification using deep learning. Upload any audio file and instantly identify its genre with our 4-stage CNN model.",
  keywords: "music genre classification, deep learning, CNN, audio analysis, machine learning",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <div className="bg-animated" />
        {children}
      </body>
    </html>
  );
}
